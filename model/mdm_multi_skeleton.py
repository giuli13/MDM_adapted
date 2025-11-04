import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
import itertools
import random
def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class GMDM(nn.Module):
    def __init__(self, modeltype, max_joints, feature_len, translation, pose_rep,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='truebones', t5_out_dim = 512, root_input_feats=7,
                 arch='trans_enc', emb_trans_dec=False, masked_attn_from_step = -1, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.max_joints = max_joints
        self.feature_len = feature_len
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.feature_len
        self.root_input_feats = root_input_feats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.skip_t5=kargs.get('skip_t5', False)
        self.value_emb=kargs.get('value_emb', False)
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, skip_t5=self.skip_t5)
        self.emb_trans_dec = emb_trans_dec
        self.masked_attn_from_step = masked_attn_from_step
        

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers, value_emb=self.value_emb)
            
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
        
        self.output_process = OutputProcess(self.data_rep,self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def encode_joints_names(self, joints_names): # joints names should be padded with None to be of max_len 
        bs = len(joints_names)
        all_names = list(itertools.chain(*joints_names))
        # raw_text - list (batch_size length) of strings with input text prompts
        names_tokens = self.t5_conditioner.tokenize(all_names)
        embs = self.t5_conditioner(names_tokens)
        embs = embs.reshape(bs, self.max_joints, embs.shape[-1])
        return embs
    

    def forward(self, x, timesteps, get_activations=False, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        joints_mask = y['joints_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)
        
        bs, njoints, nfeats, nframes = x.shape
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        # force_mask = y.get('uncond', False)
        # joints_embedded_names = self.encode_joints_names(y['joints_names'])
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind']) # applies linear layer on each frame to convert it to latent dim
        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, nframes + 1, self.num_heads, 1, 1).reshape(-1,self.num_heads, njoints, njoints)
        temporal_mask = 1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1).reshape(-1, nframes + 1, nframes + 1).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9
        output = self.seqTransDecoder(tgt=x, timesteps_embs=timesteps_emb, memory=None, spatial_mask=spatial_mask, temporal_mask = temporal_mask, y=y, get_activations=get_activations)
        if get_activations:
            activations = output[1]
            output=output[0]
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        if get_activations:
            return output, activations
        return output


    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)

class TemporalSpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, max_depth=50):
        super(TemporalSpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, max_depth, d_model)
        position_x = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, max_depth, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model/2, 2).float() * (-np.log(10000.0) / (d_model/2)))
        d_model_div_2 = int(d_model/2)
        pe[:, :, :d_model_div_2][:, :,  0::2] = torch.sin(position_x * div_term).unsqueeze(1).repeat(1, max_depth, 1)
        pe[:, :, :d_model_div_2][:, :,  1::2] = torch.cos(position_x * div_term).unsqueeze(1).repeat(1, max_depth, 1)
        pe[:, :, d_model_div_2:][:, :,  0::2] = torch.sin(position_y * div_term).unsqueeze(0).repeat(max_len, 1, 1)
        pe[:, :, d_model_div_2:][:, :,  1::2] = torch.cos(position_y * div_term).unsqueeze(0).repeat(max_len, 1, 1)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, y=None):
        # make embeddings relatively larger
        parents_depths = y['parents_depths']
        x = x + self.dropout(self.pe[:x.shape[0],:, parents_depths].squeeze(1))
        return x


# In practice, temporal attention will be applied on a window around the current frame, will be done using mask 
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('tpe', pe)

    def forward(self, x):
        # not used in the final model
        # x.shape = (frames, bs, njoints, feature_len)
        x = x + self.tpe[:x.shape[0], :].unsqueeze(1)
        return self.dropout(x)

# In practice, temporal attention will be applied on a window around the current frame, will be done using mask 
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, temporal_pos_encoder, heads):
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_pos_encoder = temporal_pos_encoder
        self.heads = heads
        self.time_embed_dim = self.latent_dim // self.heads
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def forward(self, timesteps):
        ts_emb =  self.time_embed(self.temporal_pos_encoder.tpe[timesteps]).permute(1, 0, 2)
        return ts_emb.repeat(1, 1, self.heads)


# in the case of GMDM, the input process is as follows: 
# embed each joint of each frame of each motion in batch by the same MLP, separately ! 
class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, root_input_feats, latent_dim, t5_output_dim, skip_t5=False):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.skip_t5=skip_t5
        if not self.skip_t5:
            self.joints_names_dropout = nn.Dropout(p=0.1)
            self.text_embedding = nn.Linear(t5_output_dim, self.latent_dim)
    def forward(self, x, tpos_first_frame, joints_embedded_names, crop_start_ind):
        # x.shape = [batch_size, joints, 13, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        tpos_all_joints_except_root = self.tpos_joint_embedding(tpos_first_frame[:, :, 1:, :])
        tpos_root_data = self.tpos_root_embedding(tpos_first_frame[:, :, 0:1, :self.root_input_feats])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:, :])
        root_data = self.root_embedding(x[:, :, 0:1, :self.root_input_feats])
        tpos_embedded = torch.cat([tpos_root_data, tpos_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2) 
        x = torch.cat([tpos_embedded, x_embedded], dim=0)
        if not self.skip_t5:
            joints_embedded_names = self.text_embedding(self.joints_names_dropout(joints_embedded_names.to(x.device)))
            x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        positions[:,1:,:] = positions[:,1:,:] + crop_start_ind.to(x.device).view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)

class OutputProcess(nn.Module):
    def __init__(self, data_rep, feature_len, root_feature_len, max_joints, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.root_feature_len = root_feature_len
        self.root_dembedding = nn.Linear(self.latent_dim, self.root_feature_len)
        self.joint_dembedding = nn.Linear(self.latent_dim, self.feature_len)

    def forward(self, output):
        # output shape [frames, batch_size, joints, latent_dim]
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', 'truebones_tensor']:
            root_data = self.root_dembedding(output[:, :, 0])
            zero_padding =  torch.zeros(root_data.size(0), root_data.size(1),self.feature_len - self.root_feature_len, device=output.device)
            root = torch.cat([root_data, zero_padding], dim=-1).unsqueeze(2)
            all_joints = self.joint_dembedding(output[:, :, 1:])
            output = torch.cat([root, all_joints], dim=-2)
        else:
            raise ValueError
        output = output.permute(1, 2, 3, 0)[..., 1:]  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output