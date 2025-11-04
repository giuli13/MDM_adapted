import torch
import torch.nn as nn
import clip
from model.mdm import MDM
from model.mdm_multi_skeleton import create_sin_embedding


class MDMBaseline(MDM):
    def __init__(self, *args, concat_tpos_first_frame=False, linear_layer_per_object_type=False, cond_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.arch != 'trans_enc':
            raise NotImplementedError("only transformer encoder architecture is supported for mdm-baseline")
        self.concat_tpos_first_frame = concat_tpos_first_frame
        self.linear_layer_per_object_type = linear_layer_per_object_type
        if self.concat_tpos_first_frame:
            print('CONCAT TPOS FIRST FRAME')
        if self.linear_layer_per_object_type:
            print('LINEAR LAYER PER OBJECT TYPE')
            assert cond_dict is not None
            self.input_process = InputProcessPerObjectType(self.data_rep, self.input_feats, self.latent_dim, self.nfeats, cond_dict)
            self.output_process = OutputProcessPerObjectType(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats, cond_dict)

        # embed object type
        self.embed_object_type = nn.Linear(self.clip_dim, self.latent_dim)
        print('EMBED OBJECT TYPE')
        print('Loading CLIP...')
        clip_version = kwargs.get('clip_version', None)
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)



    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert y is not None
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
       
        object_types = y['object_type']
        
        enc_obj_type = self.encode_object_type(object_types)  # [bs, d]
        emb += self.embed_object_type(self.mask_cond(enc_obj_type, force_mask=force_mask))

        if self.concat_tpos_first_frame:
            tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(-1)
            x = torch.cat((tpos_first_frame, x), dim=-1) # [bs, njoints, nfeats, nframes+1]

        if self.linear_layer_per_object_type:
            x = self.input_process(x, object_types)
        else:
            x = self.input_process(x) # [seqlen, bs, d] or [seqlen+1, bs, d]

        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, d] or [seqlen+2, bs, d]

        positions = torch.arange(nframes + 1, device=x.device).view(-1, 1, 1).expand(-1, bs, -1).clone()
        positions[1: ,:, :] += y['crop_start_ind'].to(x.device).view(1, -1, 1)
        pos_embs = create_sin_embedding(positions, self.latent_dim) # positional (frame) embedding
        
        if not self.concat_tpos_first_frame:
            pos_embs = pos_embs[1:] # remove the tpos embedding (position 0)
        
        xseq[1:] += pos_embs
        output = self.seqTransEncoder(xseq) # , src_key_padding_mask=~maskseq)

        if self.concat_tpos_first_frame:
            output = output[2:]  # [seqlen, bs, d]
        else:
            output = output[1:]  # [seqlen, bs, d]

        if self.linear_layer_per_object_type:
            output = self.output_process(output, object_types)
        else:
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def encode_object_type(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 16 if self.dataset in ['truebones'] else None  # Specific hardcoding for truebones dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()


class InputProcessPerObjectType(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, nfeats, truebones_cond_dict):
        super().__init__()
        assert data_rep == 'truebones_tensor'
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.poseEmbeddingDict = nn.ModuleDict()
        for object_type in truebones_cond_dict.keys():
            num_joints = truebones_cond_dict[object_type]['parents'].shape[0]
            input_feats = num_joints * nfeats
            self.poseEmbeddingDict[object_type] = nn.Linear(input_feats, self.latent_dim)

    def forward(self, x, object_types):
        bs, njoints, nfeats, nframes = x.shape
        assert len(object_types) == bs
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        
        _x = torch.zeros(nframes, bs, self.latent_dim, device=x.device)
        for i, object_type in enumerate(object_types):
            poseEmbedding = self.poseEmbeddingDict[object_type]
            object_input_feats = poseEmbedding.in_features
            _x[:, i, :] = self.poseEmbeddingDict[object_type](x[:, i, :object_input_feats])
        x = _x
        # x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcessPerObjectType(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, truebones_cond_dict):
        super().__init__()
        assert data_rep == 'truebones_tensor'
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        self.poseFinalDict = nn.ModuleDict()
        for object_type in truebones_cond_dict.keys():
            num_joints = truebones_cond_dict[object_type]['parents'].shape[0]
            input_feats = num_joints * nfeats
            self.poseFinalDict[object_type] = nn.Linear(self.latent_dim, input_feats)



    def forward(self, output, object_types):
        nframes, bs, d = output.shape
        assert len(object_types) == bs
        _output = torch.zeros(nframes, bs, self.input_feats, device=output.device)
        for i, object_type in enumerate(object_types):
            poseFinal = self.poseFinalDict[object_type]
            object_input_feats = poseFinal.out_features
            _output[:, i, :object_input_feats] = self.poseFinalDict[object_type](output[:, i, :])
        output = _output
        # output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
