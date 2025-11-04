# based on https://github.com/PeizhuoLi/ganimator/blob/main/fix_contact.py 

from from_ganimator.bvh.bvh_parser import BVH_file
from os.path import join as pjoin
import os
import numpy as np
import torch
from from_ganimator.models.contact import constrain_from_contact
from from_ganimator.models.kinematics import InverseKinematicsJoint2
from from_ganimator.models.transforms import repr6d2quat
from tqdm import tqdm
import argparse
import re


def continuous_filter(contact, length=2):
    contact = contact.copy()
    for j in range(contact.shape[1]):
        c = contact[:, j]
        t_len = 0
        prev = c[0]
        for i in range(contact.shape[0]):
            if prev == c[i]:
                t_len += 1
            else:
                if t_len <= length:
                    c[i - t_len:i] = c[i]
                t_len = 1
                prev = c[i]
    return contact


def fix_negative_height(contact, constrain, cid):
    floor = -1
    constrain = constrain.clone()
    for i in range(constrain.shape[0]):
        for j in range(constrain.shape[1]):
            if constrain[i, j, 1] < floor:
                constrain[i, j, 1] = floor
    return constrain


def fix_contact(bvh_file, contact, threshold=0.6):
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    cid = bvh_file.skeleton.contact_id
    glb = bvh_file.joint_position()
    rotation = bvh_file.get_rotation(repr='repr6d').to(device)
    position = bvh_file.get_position().to(device)
    contact = contact > threshold
    # contact = continuous_filter(contact)
    constrain = constrain_from_contact(contact, glb, cid)
    constrain = fix_negative_height(contact, constrain, cid).to(device)
    cid = list(range(glb.shape[1]))  # all joints
    ik_solver = InverseKinematicsJoint2(rotation, position, bvh_file.skeleton.offsets.to(device), bvh_file.skeleton.parent,
                                        constrain[:, cid], cid, 0.1, 0.01, use_velo=True)

    loop = tqdm(range(500))
    for i in loop:
        loss = ik_solver.step()
        loop.set_description(f'loss = {loss:.07f}')

    return repr6d2quat(ik_solver.rotations.detach()), ik_solver.get_position()

def create_contact_file(npy_path, contact_path, contact_idx):
    # name_split = os.path.basename(name_split).split('_')
    # animal = name_split[0]
    # rep = name_split[2]
    # num = name_split[3][1:]
    # npy_name = f'rep_{rep}_object_type_{animal}_#{num}'
    assert os.path.exists(npy_path), f'cannot generate a contact file without an npy file: {npy_path} do not exist'
    
    motion = np.load(npy_path)    
    contact = motion[..., contact_idx, -1]   
    np.save(contact_path, contact)
    
def get_contact_joints(bvh_path, zoo_attr):
    animal = os.path.basename(bvh_path).split('_')[0]
    contact_idx = [i for i,joint_name in enumerate(zoo_attr[animal]['joints_names']) if re.search(r"foot|toe|phalanx|hoof|ashi", joint_name, re.IGNORECASE)]
    
    # enrich contact_idx list with children of contacts
    extended_list = contact_idx.copy() # must use a seperate list to refrain from iterating over added elements
    for i in contact_idx:
        extended_list.extend(recursive_children(zoo_attr[animal]['parents'], i))
    
    # for debug purposes
    if len(set(extended_list) - set(contact_idx)) > 0:
        print(f'{animal}: Added {len(set(extended_list) - set(contact_idx))} children to the contact list')
        
    contact_idx = list(set(extended_list))  # remove duplicates
    
    contact_names = [zoo_attr[animal]['joints_names'][i] for i in contact_idx]
    return contact_names, contact_idx

def recursive_children(parents, idx):
    res = []
    c = np.where(parents == idx)[0]
    for i in c:
        res += [i]
        res.extend(recursive_children(parents, i))
    return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--bvh', type=str, default=None)
    parser.add_argument('--npy', type=str, default=None)
    parser.add_argument('--regen_contact', action='store_true', help="regenerate the contacts file even if it already exists.")
    args = parser.parse_args()
    zoo_attr_path = '/a/home/cc/students/cs/sigalraab/python/multi-skeleton-mdm-delete_after_250331/dataset/truebones/zoo/processed_bvh_rots/cond.npy'
    zoo_attr = np.load(zoo_attr_path, allow_pickle=True).item()
    bvh_dir_path = args.bvh if args.bvh is not None else pjoin(args.prefix, args.name + '.bvh')
    npy_dir_path = args.npy if args.npy is not None else pjoin(args.prefix, args.name + '.npy')
    bvh_files = [pjoin(bvh_dir_path, f_path) for f_path in os.listdir(bvh_dir_path) if f_path.endswith('.bvh')]
    for f_path in bvh_files:
        contact_names, contact_idx = get_contact_joints(f_path, zoo_attr)
        
        if len(contact_names) == 0:
            print(f'No contact joints found in {os.path.basename(f_path)}')
            continue
        contact_path = f_path + '.contact.npy'   
        if f_path.endswith('_fixed.bvh'):
            npy_path =  pjoin(npy_dir_path, os.path.basename(f_path)[:-10]+'.npy')
        else:     
            npy_path =  pjoin(npy_dir_path, os.path.basename(f_path)[:-4]+'.npy')
        if os.path.exists(npy_path):
            create_contact_file(npy_path, contact_path, contact_idx)
        
        contact = np.load(contact_path)
        bvh_file = BVH_file(f_path, no_scale=True, requires_contact=True, 
                            joint_reduction=False, contact_names=contact_names)

        res = fix_contact(bvh_file, contact, args.threshold)

        fixed_bvh_path = f_path.replace('.bvh', f'_fixed_th{args.threshold}.contact.bvh')
        bvh_file.writer.write(fixed_bvh_path, res[0], res[1], 
                            names=bvh_file.skeleton.names, repr='quat', frametime=bvh_file.frametime, order='xyz')
