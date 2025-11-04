import torch
from utils.parents_traverse import depth_list, skeleton_mask, first_half_pe, second_half_pe


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def n_joints_to_mask(n_joints, max_joints):
    mask = torch.arange(max_joints + 1, device=n_joints.device).expand(len(n_joints), max_joints + 1) < (n_joints.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    return mask

def length_to_temp_mask(max_len_mask, lengths, max_len):
    mask = torch.arange(max_len + 1, device=lengths.device).expand(len(lengths), max_len + 1) < (lengths.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float() 
    mask = mask.logical_and(max_len_mask)
    return mask

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def create_padded_adj_mask(adj_mask_np, max_joints, n_joints, includ_root = False):
        adj_mask = torch.from_numpy(adj_mask_np)
        if includ_root:
            adj_mask[:, 0] = 1.0 # include root joint at spatial attn for all joints 
        padded_adj_mask = torch.zeros((max_joints+ 1, max_joints + 1)) 
        padded_adj_mask[1:n_joints + 1, 1:n_joints + 1] = adj_mask
        padded_adj_mask[0, :n_joints + 1] =  1.0
        padded_adj_mask[1:n_joints + 1, 0] =  1.0
        return padded_adj_mask

def create_padded_relation(relation_np, max_joints, n_joints):
        # it counts on spatial attention masks!
        relation = torch.from_numpy(relation_np)
        padded_relation = torch.zeros((max_joints, max_joints)) 
        padded_relation[:n_joints, :n_joints ] = relation
        return padded_relation

def truebones_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    tposfirstframebatch = [b['tpos_first_frame'] for b in notnone_batches]
    meanbatch = [b['mean'] for b in notnone_batches]
    stdbatch = [b['std'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
        
    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]
    if 'crop_start_ind' in notnone_batches[0]:
        cropstartindbatch = [b['crop_start_ind'] for b in notnone_batches]
        
    
    
    databatchTensor = collate_tensors(databatch)
    tposfirstframebatchTensor = collate_tensors(tposfirstframebatch)
    meanbatchTensor = collate_tensors(meanbatch)
    stdbatchTensor = collate_tensors(stdbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    cropstartindTensor = torch.as_tensor(cropstartindbatch)
    lengthsmaskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    collated_temporalmasksbatch = collate_tensors(temporalmasksbatch)
    maskbatchTensor = length_to_temp_mask(collated_temporalmasksbatch, lenbatchTensor, collated_temporalmasksbatch[0].size(0) - 1).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'lengths_mask': lengthsmaskbatchTensor, 'tpos_first_frame': tposfirstframebatchTensor, 'crop_start_ind': cropstartindTensor, 'mean': meanbatchTensor, 'std':stdbatchTensor}}
    
    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})
    
    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})
          
    if 'joints_names_embs' in notnone_batches[0]:
        jointsnamesembsbatch = [b['joints_names_embs'] for b in notnone_batches]
        jointsnamesembsbatchTensor = collate_tensors(jointsnamesembsbatch)
        cond['y'].update({'joints_names_embs': jointsnamesembsbatchTensor})
        
    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    if 'bones_len' in notnone_batches[0]:
        boneslenbatch = [b['bones_len'] for b in notnone_batches]


    cond['y'].update({'joints_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})
    cond['y'].update({'bones_len': torch.stack(boneslenbatch)})

    return motion, cond


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches] #smpl n_joints 
    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]
    else:
        temporalmasksbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    
    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})
    
    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})
    
    if 'parents' in notnone_batches[0]:
        jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        # tempmaskbatchTensor = collate_tensors(temporalmasksbatch).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        tempmaskbatchTensor = length_to_temp_mask(collate_tensors(temporalmasksbatch), lenbatchTensor, temporalmasksbatch[0].size(0) - 1).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        parentsbatch = [b['parents'] for b in notnone_batches]
        parentsdepthbatch = [depth_list(b['parents']) for b in notnone_batches]
        parentsmasksbatch = [skeleton_mask(b['parents']) for b in notnone_batches]

        if 'joints_relations' in notnone_batches[0]:
            jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]
        else:
            jointsrelationsbatch = [torch.tensor(0) for b in notnone_batches]
        if 'graph_dist' in notnone_batches[0]:
            graphdistbatch = [b['graph_dist'] for b in notnone_batches]
        else:
            graphdistbatch = [torch.tensor(0) for b in notnone_batches]
        if 'bones_len' in notnone_batches[0]:
            boneslenbatch = [b['bones_len'] for b in notnone_batches]
        else:
           boneslenbatch = [torch.tensor(0) for b in notnone_batches]
        if 'joints_perm' in notnone_batches[0]:
            jointspermbatch = [b['joints_perm'] for b in notnone_batches]
        else:
            jointspermbatch = [torch.tensor(0) for b in notnone_batches]
        if 'joints_perm_inv' in notnone_batches[0]:
            invjointspermbatch = [b['joints_perm_inv'] for b in notnone_batches]
        else:
            invjointspermbatch = [torch.tensor(0) for b in notnone_batches]

        cond['y'].update({'parents': torch.tensor(parentsbatch).float()})
        cond['y'].update({'parents_masks': torch.stack(parentsmasksbatch)})
        cond['y'].update({'parents_depths': torch.tensor(parentsdepthbatch)})
        cond['y'].update({'joints_mask': jointsmaskbatchTensor})
        cond['y'].update({'temp_mask': tempmaskbatchTensor})
        cond['y'].update({'n_joints': jointsnumbatchTensor})
        cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
        cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})
        cond['y'].update({'bones_len': torch.stack(boneslenbatch)})
        cond['y'].update({'joints_perm': torch.stack(jointspermbatch)})
        cond['y'].update({'joints_perm_inv': torch.stack(invjointspermbatch)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


""" recieves list of tuples of the form: 
(word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), parents, max_joints, temp_mask_template, object_type, adj_mask)
"""
def t2m2p_collate(batch):
    adapted_batch = []
    for b in batch:        
        n_joints = b[4].shape[1]
        max_jojnts = b[8]
        motion = torch.zeros((b[4].shape[0], max_jojnts, b[4].shape[3])) # (frames, max_joints, feature_len)
        motion[:, :b[4].shape[1], :] = torch.tensor(b[4].sum(axis=2))
        parents = [-1] * (max_jojnts - n_joints) #make parents of length max_joints, pad irrelevant joint with -1
        parents = b[7] + parents
        temporal_mask = b[9][:b[4].shape[0] + 1, :b[4].shape[0] + 1].clone()
        padded_adj_mask = create_padded_adj_mask(b[11], max_jojnts, n_joints)

        item = {
            'inp':  motion.permute(1, 2, 0).float(), # [seqlen, J, 31] -> [J, 31,  seqlen]
            'text': b[2],
            'tokens': b[6],
            'lengths': b[5],
            'parents': parents,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'object_type': b[10], # object name only will be used for first experiment
            'n_joints': n_joints,
            'temporal_mask' : temporal_mask,
            'adj_mask' : padded_adj_mask,
        } 
        adapted_batch.append(item)

    return collate(adapted_batch)

""" recieves list of tuples of the form: 
word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), parents, max_joints, self.temporal_mask_template, graph_dist, joints_relations, bones_len, joints_perm, joints_perm_inv
"""
def t2m_mixed_collate(batch):
    max_joints = 143 
    adapted_batch = []
    for b in batch:        
        n_joints = b[4].shape[1]
        motion = torch.zeros((b[4].shape[0], max_joints, b[4].shape[2])) # (frames, max_joints, feature_len)
        motion[:, :b[4].shape[1], :] = torch.tensor(b[4])
        parents = b[7] + [0 for i in range(max_joints - n_joints)]
        temporal_mask = b[8][:b[4].shape[0] + 1, :b[4].shape[0] + 1].clone()
        padded_joints_relations =  create_padded_relation(b[10], max_joints, n_joints, b[10].max() + 1)
        padded_graph_dist =  create_padded_relation(b[9], max_joints, n_joints, b[9].max() + 1)
        padded_bones_len =  create_padded_relation(b[11], max_joints, n_joints, b[11].max() + 1)
        joints_perm = torch.tensor(b[12] + [n_joints + i for i in range(max_joints - n_joints)])
        joints_perm_inv = torch.tensor(b[13] + [n_joints + i for i in range(max_joints - n_joints)])
        item = {
            'inp':  motion.permute(1, 2, 0).float(), # [seqlen, J, 25] -> [J, 25,  seqlen]
            'text': b[2],
            'tokens': b[6],
            'lengths': b[5],
            'parents': parents,
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'bones_len' : padded_bones_len,
            'joints_perm': joints_perm,
            'joints_perm_inv': joints_perm_inv
        } 
        adapted_batch.append(item)

    return collate(adapted_batch)

""" recieves list of tuples of the form: 
 motion, m_length, parents, joints_perm, inv_joints_perm, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, normalized_edge_len, object_type, joints_names_embs, ind, mean, std
"""
def truebones_mixed_collate(batch):
    max_joints = 143 
    adapted_batch = []
    for b in batch:  
        max_len, n_joints, n_feats = b[0].shape
        tpos_first_frame = torch.zeros((max_joints, n_feats))
        tpos_first_frame[:n_joints] = torch.tensor(b[3])
        motion = torch.zeros((max_len, max_joints, n_feats)) # (frames, max_joints, feature_len) 
        motion[:, :b[0].shape[1], :] = torch.tensor(b[0])   
        joints_names_embs = torch.zeros((max_joints, b[10].shape[1]))
        joints_names_embs[:n_joints] = torch.tensor(b[10])
        crop_start_ind = b[11]
        mean = torch.zeros((max_joints, n_feats))
        mean[:n_joints] = torch.tensor(b[12])
        std = torch.ones((max_joints, n_feats))
        std[:n_joints] = torch.tensor(b[13])
        n_joints = b[0].shape[1]
        temporal_mask = b[5][:max_len + 1, :max_len + 1].clone()
        
        padded_joints_relations =  create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[6], max_joints, n_joints)
        padded_bones_len =  create_padded_relation(b[8], max_joints, n_joints - 1)
        object_type = b[9]

        item = {
            'inp': motion.permute(1, 2, 0).float(), # [seqlen , J, 13] -> [J, 13,  seqlen]
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'bones_len' : padded_bones_len,
            'object_type': object_type,
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame, 
            'crop_start_ind': crop_start_ind,
            'mean': mean,
            'std': std
        } 
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)