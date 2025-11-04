import torch 
import torch.nn as nn
import math
from torch.autograd import Variable
from typing import Optional, Union, Callable
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
CUDA_LAUNCH_BLOCKING=1



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
        first_half_pe = y["first_half_pe"].long()
        second_half_pe = y["second_half_pe"].long()
        first_half = Variable(self.pjpe[first_half_pe, :], requires_grad = False) # bs, max_joints, d_model
        second_half = Variable(self.pjpe[second_half_pe ,:], requires_grad = False) # bs, max_joints, d_model
        add_pe = torch.cat((first_half, second_half), dim = -1) 
        if self.heads != -1:
            add_pe = add_pe.repeat(1, 1, self.heads)
        x = x + add_pe.unsqueeze(0)
        return self.dropout(x)

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
        pe = pe.unsqueeze(1).unsqueeze(1)

        self.register_buffer('tpe', pe)

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)
        add_tpe = Variable(self.tpe[:x.shape[0],:,:, :], requires_grad = False)
        x = x + add_tpe
        return self.dropout(x)

class MotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None): 
        super().__init__(decoder_layer, num_layers, norm)

    # TODO: implement cross attention using memory when needed
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, y=None) -> Tensor:
        output = tgt
        for layer_ind, mod in enumerate(self.layers):
            output, attn_scores = mod(
                    output, timesteps_embs, layer_ind, spatial_mask, temporal_mask, 
                    tgt_key_padding_mask, memory_key_padding_mask, y)
        if self.norm is not None:
            output = self.norm(output)

        return output, attn_scores
        
class MotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, per_head_spatial_pe=False):
        
        super(MotionDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.heads = nhead
        # multi head attention
        self.spatial_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # positional encoding 
        if not per_head_spatial_pe:
            self.spatial_pe = ParentJointEncoding(d_model, use_heads=-1) # not per head pe !
        else:
            self.spatial_pe = ParentJointEncoding(d_model, use_heads=self.heads) #  per head pe !
        self.temporal_pe = TemporalPositionalEncoding(d_model)

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, timesteps_embs: Tensor, layer_ind: int, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        if layer_ind == 0:
            x_add_pe = self.spatial_pe(x, y)
            concat_spatial = torch.cat((timesteps_embs.unsqueeze(2).repeat(frames, 1, 1, 1), x_add_pe), axis=2)
        else:
            concat_spatial = torch.cat((timesteps_embs.unsqueeze(2).repeat(frames, 1, 1, 1), x), axis=2)
        concat_spatial = concat_spatial.transpose(0, 2).reshape(njoints + 1, bs * frames, feats)  
        attn_output, attn_scores = self.spatial_attn(concat_spatial, concat_spatial, concat_spatial,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        attn_output = attn_output.reshape(njoints+1, bs, frames, feats)[1:].transpose(0,2)
        attn_scores = attn_scores.reshape(bs, frames, njoints+1, njoints+1)
        return self.dropout1(attn_output), attn_scores

    # temporal attention block
    def _temporal_mha_block(self, x: Tensor, timesteps_embs: Tensor, layer_ind: int, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        if layer_ind == 0:
            x_add_pe = self.temporal_pe(x)
            concat_temporal = torch.cat((timesteps_embs.unsqueeze(2).repeat(1, 1, njoints, 1), x_add_pe), axis=0) 
        else:
            concat_temporal = torch.cat((timesteps_embs.unsqueeze(2).repeat(1, 1, njoints, 1), x), axis=0) 
        concat_temporal = concat_temporal.reshape(frames + 1, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(concat_temporal, concat_temporal, concat_temporal,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.reshape(frames + 1, bs ,njoints, feats)[1:]
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def forward(self,
        tgt: Tensor,
        timesteps_embs: Tensor,
        layer_ind: int,
        spatial_mask: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None) -> Tensor:
        x = tgt
        spatial_attn_output, attn_scores = self._spatial_mha_block(x, timesteps_embs, layer_ind, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block(x, timesteps_embs, layer_ind, temporal_mask, tgt_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x, attn_scores
    
######################################################################################################################################

# add timestep at the beggining only 
# 
class FullMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=10): 
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)
        
        d_model = decoder_layer.d_model
        self.topology_key_emb = nn.Embedding(max_path_len + 3, d_model) # 0-(max_path_len-1) / max_path_len -> unrechable/ max_path_len + 1 -> ts_token_conn/ max_path_len + 2 -> far (>max_path_len)
        self.edge_key_emb = nn.Embedding(6, d_model) # 0->parent/1->child/2->self/3->ts_token_edge
        self.topology_query_emb = nn.Embedding(max_path_len + 3, d_model) # 0-(max_path_len-1) / max_path_len -> unrechable/ max_path_len + 1 -> ts_token_conn/ max_path_len + 2 -> far (>max_path_len)
        self.edge_query_emb = nn.Embedding(6, d_model) # 0->parent/1->child/2->self/3->ts_token_edge
        self.topology_value_emb = nn.Embedding(max_path_len + 3, d_model) # 0-(max_path_len-1) / max_path_len -> unrechable/ max_path_len + 1 -> ts_token_conn/ max_path_len + 2 -> far (>max_path_len)
        self.edge_value_emb = nn.Embedding(6, d_model) # 0->parent/1->child/2->self/3->ts_token_edge
        
        
    # TODO: implement cross attention using memory when needed
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, y=None) -> Tensor:
        topology_rel = y['topology_relation'].long().to(tgt.device)
        edge_rel = y['edge_relation'].long().to(tgt.device)
        output = tgt
        # concatenate timestep emb token as additional joint
        frames = output.shape[0]
        output = torch.cat((timesteps_embs.unsqueeze(2).repeat(frames, 1, 1, 1), output), axis=2) # frames, bs, njoints, feats
        for layer_ind, mod in enumerate(self.layers):
            output = mod(
                    output, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, self.edge_value_emb, self.topology_key_emb, self.topology_query_emb, self.topology_value_emb, spatial_mask, temporal_mask, 
                    tgt_key_padding_mask, memory_key_padding_mask, y)
        if self.norm is not None:
            output = self.norm(output)
        # remove timestep emb token from joints dimension
        output = output[:, :, 1:]
        return output

class FullMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, temp_attn_dim=None):
        super(FullMotionDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model= d_model
        self.heads = nhead
        if temp_attn_dim is None:
            self.temp_dim = d_model//8
        else:
            self.temp_dim = temp_attn_dim
        
        self.spatial_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.enc_temp = nn.Linear(d_model, self.temp_dim)
        # self.dec_temp = nn.Linear(self.temp_dim, d_model)
        #self.temporal_attn = MultiheadAttention(self.temp_dim * 23, nhead, dropout=dropout) # for generalized verse, 143 instead 22
        self.temporal_attn = MultiheadAttention(self.d_model, nhead, dropout=dropout) # for generalized verse, 143 instead 22

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb, 
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        head_dim = self.d_model // self.heads
        scaling = float(head_dim) ** -0.5
        topology_rel = topology_rel.to(x.device)
        edge_rel = edge_rel.to(x.device)
        x = x.transpose(0, 2).reshape(njoints, bs * frames, feats)  
        emb_topo_key = topology_key_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) # (1, heads, topo_types, head_dim)
        emb_topo_query = topology_query_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) #(1, heads, topo_types, head_dim)
        emb_topo_value = topology_value_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) #(1, heads, topo_types, head_dim)
        emb_edge_key = edge_key_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) #(1, heads, topo_types, head_dim)
        emb_edge_query = edge_query_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) #(1, heads, topo_types, head_dim)
        emb_edge_value = edge_value_emb.weight.view(1, -1, self.heads, head_dim).transpose(1, 2) #(1, heads, topo_types, head_dim)
        q, k, v = F.linear(x, self.spatial_attn.in_proj_weight).chunk(3, dim=-1) # each (njoints+1, frames * bs, d_model)
        q = q * scaling
        q = q.contiguous().view(njoints, bs * frames, self.heads, head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(njoints, bs * frames, self.heads, head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(njoints, bs * frames, self.heads, head_dim).permute(1, 2, 0, 3)
        # get topology & edge biases
        topo_key_attn = torch.matmul(k, emb_topo_key.transpose(2, 3)) 

        topo_key_attn = torch.gather(
            topo_key_attn, 3, topology_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1)
        )
        topo_query_attn = torch.matmul(q, emb_topo_query.transpose(2, 3))
        topo_query_attn = torch.gather(
            topo_query_attn, 3, topology_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1)
        )
        edge_key_attn = torch.matmul(k, emb_edge_key.transpose(2, 3))
        edge_key_attn = torch.gather(
            edge_key_attn, 3, edge_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1)
        )
        edge_query_attn = torch.matmul(q, emb_edge_query.transpose(2, 3))
        edge_query_attn = torch.gather(
            edge_query_attn, 3, edge_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1)
        )

        topo_bias = topo_key_attn + topo_query_attn
        edge_bais = edge_key_attn + edge_query_attn
        attn_mask = (attn_mask + topo_bias + edge_bais).reshape(-1, njoints, njoints)
        _, attn_scores = self.spatial_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        attn_scores = attn_scores.unsqueeze(1).repeat(1, self.heads, 1, 1)
        value_topo_att = torch.zeros(
            (bs*frames, self.heads, njoints, emb_topo_value.shape[2]), 
            device=emb_topo_value.device
        )
        value_topo_att = torch.scatter_add(
            value_topo_att, 3, topology_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1), attn_scores
        )
        value_edge_att = torch.zeros(
            (bs*frames, self.heads, njoints, emb_edge_value.shape[2]),
            device=emb_edge_value.device,
        )
        value_edge_att = torch.scatter_add(
            value_edge_att, 3, edge_rel.unsqueeze(1).repeat(frames, self.heads, 1, 1), attn_scores
        )

        value_topo = torch.matmul(value_topo_att, emb_topo_value)
        value_edge = torch.matmul(value_edge_att, emb_edge_value)
        value_bias = (value_topo + value_edge).permute(2, 0, 1, 3).reshape(njoints, frames * bs, self.heads * head_dim)

        attn_output, _ = self.spatial_attn(x, x, x + value_bias,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)

        attn_output = attn_output.reshape(njoints, bs, frames, feats).transpose(0,2)
        return self.dropout1(attn_output)

    # # temporal attention block
    # def _temporal_mha_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    #     attn_mask_ = attn_mask[..., 1:, 1:]
    #     x_enc = self.enc_temp(x)
    #     frames, bs, njoints, feats= x_enc.size() 
    #     x_enc = x_enc.reshape(frames, bs, njoints*feats)
    #     output_attn, output_scores = self.temporal_attn(x_enc, x_enc, x_enc,
    #                             attn_mask=attn_mask_,
    #                             key_padding_mask=key_padding_mask)
    #     output_attn = output_attn.view(frames, bs ,njoints, feats)
    #     output_dec = self.dec_temp(output_attn)
    #     return self.dropout2(output_dec)
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        attn_mask_ = attn_mask[..., 1:, 1:]
        x = x.reshape(frames, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(x, x, x,
                                attn_mask=attn_mask_,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.reshape(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def forward(self,
        tgt: Tensor,
        topology_rel: Tensor,
        edge_rel: Tensor,
        edge_key_emb,
        edge_query_emb,
        edge_value_emb,
        topo_key_emb,
        topo_query_emb,
        topo_value_emb,
        spatial_mask: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None) -> Tensor:
        x = tgt
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb, 
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        #x = self.norm2(x + self._temporal_mha_block(x, temporal_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x

######################################################################################################################################

# in contrust to torch implementation, accepts batch_size at index 0  
# in contrust to torch implementation, accepts 4 dim tensors, mo meed to union frame/joints dimension with bs
class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads):
        super().__init__()

        self.nheads = nheads

        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(nheads * att_size, d_model)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, d_k, k_len]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.nheads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.nheads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        spatial_bias = (query_hop + key_hop)
        edge_bais = (query_edge + key_edge)

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bais

        x = x * self.scale

        if mask is not None:
            # if mask.shape != x.shape:
            #     mask = mask.view(mask.shape[0], 1, 1, mask.shape[1])
            # x = x.masked_fill(
            #     mask, float("-inf")
            # )
            x = x + mask

        x = torch.softmax(x, dim=3)
        x = self.dropout(x)
        if value_hop_emb is not None:
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
            value_edge_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )

        x = torch.matmul(x, v)
        if value_hop_emb is not None:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=5, value_emb=False): 
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)
        
        self.d_model = decoder_layer.d_model
        self.topology_key_emb = nn.Embedding(max_path_len + 2, self.d_model) # 'far': max_path_len + 1
        self.edge_key_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.topology_query_emb = nn.Embedding(max_path_len + 2, self.d_model) # 'far': max_path_len + 1
        self.edge_query_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 2, self.d_model) # 'far': max_path_len + 1
            self.edge_value_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        

        
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, y=None, get_activations=False) -> Tensor:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_activations:
            activations=dict()
        # output = torch.cat((timesteps_embs.unsqueeze(2).repeat(frames, 1, 1, 1), output), axis=2) # frames, bs, njoints, feats
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            output = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask, temporal_mask, 
                    tgt_key_padding_mask, memory_key_padding_mask, y)
            if get_activations:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        # # remove timestep emb token from joints dimension
        # output = output[:, :, 1:]
        if get_activations:
            return output, activations
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, temp_attn_dim=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model= d_model
        self.heads = nhead
        self.spatial_attn = GraphMultiHeadAttention(d_model = d_model, nheads = nhead, dropout=dropout)
        self.temporal_attn = MultiheadAttention(self.d_model, nhead, dropout=dropout) 
        self.embed_timesteps = nn.Linear(d_model, d_model)

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        #x.shape (frames, bs, njoints, feature_len)
        frames, bs, njoints, feature_len = x.shape
        x = x.view(frames * bs, njoints, feature_len)
        topology_rel = topology_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        edge_rel = edge_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        
        attn_output = self.spatial_attn(x, x, x, topology_query_emb.weight, edge_query_emb.weight, topology_key_emb.weight, edge_key_emb.weight, None if topology_value_emb is None else topology_value_emb.weight, 
        None if edge_value_emb is None else edge_value_emb.weight, topology_rel, edge_rel, attn_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len) # njoints, bs, frames, feature_len
        return self.dropout1(attn_output)
    
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        # attn_mask_ = attn_mask[..., 1:, 1:]
        x = x.view(frames, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def forward(self,
        tgt: Tensor,
        timesteps_emb: Tensor,
        topology_rel: Tensor,
        edge_rel: Tensor,
        edge_key_emb,
        edge_query_emb,
        edge_value_emb,
        topo_key_emb,
        topo_query_emb,
        topo_value_emb,
        spatial_mask: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None) -> Tensor:
        x = tgt #(frames, bs, njoints, feature_len)
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x
