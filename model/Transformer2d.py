import torch 
import torch.nn as nn
import math
import copy
from os.path import join as pjoin
import os 
from utils import dist_util
from torch.autograd import Variable
import numpy as np


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.transpose(0,1).float()
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, attention_type, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model//heads
        self.h = heads
        self.attention_type = attention_type
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        d1, bs, d2, feature_len =  q.size()
        k = self.k_linear(k).view(d1, bs, d2, self.h, self.d_k)
        q = self.q_linear(q).view(d1, bs, d2, self.h, self.d_k)
        v = self.v_linear(v).view(d1, bs, d2, self.h, self.d_k)

        k = k.transpose(2, 3)
        q = q.transpose(2, 3)
        v = v.transpose(2, 3)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(2, 3).contiguous().view(d1, bs, d2, self.d_model)
        output = self.out(concat)

        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# In practice, temporal attention will be applied on a window around the current frame, will be done using mask 
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)

        self.register_buffer('tpe', pe)

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)
        add_tpe = Variable(self.tpe[:,:,:x.shape[2], :], requires_grad = False)
        x = x + add_tpe
        return self.dropout(x)

# In practice, temporal attention will be applied on a window around the current frame, will be done using mask 
class SpatialPositionalEncoding_(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(SpatialPositionalEncoding_, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('spe', pe)


    def forward(self, x, depth_map): 
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        # add constant to embedding
        pe_add = Variable(self.spe[depth_map], requires_grad=False).permute(2, 0, 1, 3) #nframes, bs, njoints, ndim

        x = x + pe_add
        return self.dropout(x)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000, max_depth=50):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, max_depth, d_model)
        position_temp = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_depth = torch.arange(0, max_depth, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / (d_model*2)))
        internal_term = torch.sqrt((torch.pow(position_temp.unsqueeze(1), 2) * torch.pow(position_depth.unsqueeze(0), 2))) * div_term
        pe[:, :,  0::2] = torch.sin(internal_term)
        pe[:, :,  1::2] = torch.cos(internal_term)
        self.register_buffer('spe', pe)

    def forward(self, x, y=None):
        parents_depths = y['parents_depths']
        pe_add = Variable(self.spe[:x.shape[2],parents_depths].permute(1, 0, 2, 3)[:, torch.arange(x.shape[2]), torch.arange(x.shape[2])].unsqueeze(0)) #bs, njoints, ndim
        x = x +  pe_add
        return x

class ParentJointEncoding(nn.Module):
    def __init__(self, d_model, use_heads=4, max_joints=143, dropout=0.1):
        super(ParentJointEncoding, self).__init__()
        self.heads = use_heads
        if use_heads == -1: #do not take num of heads in to consideration
            self.d_model = d_model // 2
        else:
            self.d_model = d_model // (self.heads * 2)

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_joints + 1, self.d_model)
        joint_index = torch.arange(0, max_joints, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:-1, 0::2] = torch.sin(joint_index * div_term)
        pe[:-1, 1::2] = torch.cos(joint_index * div_term)

        self.register_buffer('pjpe', pe)

    def forward(self, x, y=None):
        # x (frame_num, bs, n_joints, feature_len)
        frame_num, bs, n_joints, feature_len = x.size()
        parents = y["parents"].long()
        joint_ind_pe = Variable(self.pjpe[:x.shape[2], :], requires_grad = False) # max_joints, d_model
        parents_ind_pe = Variable(self.pjpe[parents ,:], requires_grad = False) # bs, max_joints, d_model
        add_pe = torch.cat((joint_ind_pe.unsqueeze(0).repeat(bs, 1, 1) , parents_ind_pe), dim = -1) 
        if self.heads != -1:
            add_pe = add_pe.repeat(1, 1, self.heads)
        x = x + add_pe.unsqueeze(0)
        return self.dropout(x)

class ParentJointEncoding2(nn.Module):
    def __init__(self, d_model, use_heads=4, max_joints=143, dropout=0.1):
        super(ParentJointEncoding2, self).__init__()
        self.heads = use_heads
        if use_heads == -1: #do not take num of heads in to consideration
            self.d_model = d_model // 2
        else:
            self.d_model = d_model // (self.heads * 2)

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_joints + 1, self.d_model)
        joint_index = torch.arange(0, max_joints, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:-1, 0::2] = torch.sin(joint_index * div_term)
        pe[:-1, 1::2] = torch.cos(joint_index * div_term)

        self.register_buffer('pjpe', pe)

    def forward(self, x, y=None):
        # x (frame_num, bs, n_joints, feature_len)
        frame_num, bs, n_joints, feature_len = x.size()
        first_half_pe = y["first_half_pe"].long()
        second_half_pe = y["second_half_pe"].long()
        first_half = Variable(self.pjpe[first_half_pe, :], requires_grad = False) # bs, max_joints, d_model
        second_half = Variable(self.pjpe[second_half_pe ,:], requires_grad = False) # bs, max_joints, d_model
        add_pe = torch.cat((first_half, second_half), dim = -1) 
        if self.heads != -1:
            add_pe = add_pe.repeat(1, 1, self.heads)
        x = x + add_pe.unsqueeze(0)
        return self.dropout(x)

# per head stpe
class STPositionalEncoding3(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, max_len=1000, max_depth=50):
        super(STPositionalEncoding3, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dk = d_model // heads
        pe = torch.zeros(max_len, max_depth, self.dk)
        position_temp = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_depth = torch.arange(0, max_depth, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dk, 2).float() * (-np.log(10000.0) / (self.dk/2)))
        internal_term = torch.sqrt((torch.pow(position_temp.unsqueeze(1), 2) * torch.pow(position_depth.unsqueeze(0), 2))) * div_term
        pe[:, :,  0::2] = torch.sin(internal_term)
        pe[:, :,  1::2] = torch.cos(internal_term)
        pe = pe.repeat(1, 1, heads) 
        self.register_buffer('stpe', pe)

    def forward(self, x, y=None):
        parents_depths = y['parents_depths']
        pe_add = Variable(self.stpe[:x.shape[2],parents_depths][torch.arange()].permute(1,0,2)) #nframes, bs, njoints, ndim
        x = x +  pe_add
        return x

class TransformerDecoder2dLayer(nn.Module):
    def __init__(self, latent_dim=512, num_heads_temporal=4, num_heads_spatial=4, dropout=0.1, **kwargs):
        super().__init__()
        self.norm_1 = Norm(latent_dim)
        self.norm_2 = Norm(latent_dim)
        self.norm_3 = Norm(latent_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.ff = FeedForward(latent_dim)

        self.num_heads_temporal = num_heads_temporal
        self.num_heads_spatial = num_heads_spatial
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.spatial_attn = MultiHeadAttention(self.num_heads_spatial, self.latent_dim, 'spatial') 
        self.temporal_attn = MultiHeadAttention(self.num_heads_temporal, self.latent_dim, 'temporal')
        #self.spatial_pe = ParentJointEncoding(self.latent_dim, use_heads=self.num_heads_spatial) #per head pe!
        #self.spatial_pe = ParentJointEncoding(self.latent_dim, use_heads=-1) # not per head pe !
        #self.spatial_pe = ParentJointEncoding2(self.latent_dim, use_heads=self.num_heads_spatial) #per head pe!
        self.spatial_pe = ParentJointEncoding2(self.latent_dim, use_heads=-1) # not per head pe !
        self.spatial_temporal_pe = STPositionalEncoding3(d_model=self.latent_dim, heads=self.num_heads_spatial)
        self.temporal_pe = TemporalPositionalEncoding(self.latent_dim)

    
    # spat_temp_pe
    def forward(self, x, y, layer_ind, timesteps_embs, spat_mask, temp_mask):
        # spatial attention + skip connection
        # adapt x to spatial attn
        # timesteps embs shape (1,bs,d_model)
        # concat emb to joints (frame_num, bs, n_joints + 1, feature_len)
        if layer_ind == 0:
            x1 = self.spatial_pe(x, y)
            x1 = torch.cat((timesteps_embs.unsqueeze(2).repeat(x1.size(0), 1, 1, 1), x1), axis=2)
        else:
            x1 = torch.cat((timesteps_embs.unsqueeze(2).repeat(x.size(0), 1, 1, 1), x), axis=2)
            
        x1 = x + self.dropout_1(self.spatial_attn(x1, x1, x1, mask=spat_mask))[:, :, 1:,:]
        x1 = self.norm_1(x1).transpose(0,2)# n_joints, bs, frame_num, feature_len 

        if layer_ind == 0:
            x1 = self.temporal_pe(x1)
        # concat emb to frames (n_joints, bs, frame_num + 1, feature_len)
        x2 = torch.cat((timesteps_embs.unsqueeze(2).repeat(x1.size(0), 1, 1, 1), x1), axis=2) 
        x2 = x1 + self.dropout_2(self.temporal_attn(x2, x2, x2, mask=temp_mask))[:, :, 1:, :]
        x2 = self.norm_2(x2).transpose(0,2) # frames , bs, n_joints, feature_len  

        x = self.norm_3(x2 + self.ff(x2)) 

        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

class TransformerDecoder2d(nn.Module):
    def __init__(self, latent_dim, N, heads, save_path=None, is_train=True, device=None,
                 save_freq=2000):
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())
        self.save_freq = save_freq
        if save_path is None:
            save_path = './trained_models/decoder/'
        self.save_path = save_path
        self.is_train = is_train
        self.epoch_count = 0
        self.optimizer = None

        self.N = N
        self.layers = get_clones(TransformerDecoder2dLayer(latent_dim, heads, heads), N)
        self.norm = Norm(latent_dim)
        # os.makedirs(pjoin(save_path, 'model'), exist_ok=True)
        # os.makedirs(pjoin(save_path, 'optimizer'), exist_ok=True)

    def forward(self, x, y, timesteps_embs, spat_mask=None, temp_mask=None):
        for i in range(self.N):
            x = self.layers[i](x, y, i, timesteps_embs, spat_mask, temp_mask)
        x = self.norm(x)
        return x
    

    def epoch(self):
        self.epoch_count += 1

    def set_optimizer(self, lr, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def save_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if epoch % self.save_freq == 0:
            torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/%05d.pt' % epoch))
            torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/%05d.pt' % epoch))

        torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/latest.pt'))
        torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/latest.pt'))

    def load_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if isinstance(epoch, str):
            state_dict = torch.load(epoch, map_location=self.device)
            self.layers.load_state_dict(state_dict)

        else:
            filename = ('%05d.pt' % epoch) if epoch != -1 else 'latest.pt'
            state_dict = torch.load(pjoin(self.save_path, f'model/{filename}'), map_location=self.device)
            self.layers.load_state_dict(state_dict)

            if self.is_train:
                state_dict = torch.load(pjoin(self.save_path, f'optimizer/{filename}'), map_location=self.device)
                self.optimizer.load_state_dict(state_dict)