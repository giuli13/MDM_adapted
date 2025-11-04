from Animation import *
import numpy as np 
from os.path import join as pjoin
from Quaternions import Quaternions
from visualizations import plot_3d_motion
from data_loaders.truebones.utils.motion_process import *
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_parents, smpl_offsets, t2m_raw_offsets
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos
""" get 6d rotations continuous representation"""
def get_6d_rep(qs):
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)

def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    
    #x_norm = np.linalg.norm(x_raw, axis=-1, keepdims=True)
    #x = np.divide(x_raw, x_norm, out=np.zeros_like(x_raw), where=x_norm!=0)
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    #z_norm = np.linalg.norm(z, axis=-1, keepdims=True) 
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    #z = np.divide(z, z_norm, out=np.zeros_like(z), where=x_norm!=0)

    y = np.cross(z, x, axis=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = np.concatenate([x, y, z], axis=-1)
    return mat

def get_motion_inner(treubones_rep_motion, parents, joints_num):
    rots, offs, parents = recover_quats_and_offs_np(treubones_rep_motion, joints_num, parents)
    new_rots = kinematic_chains_rots_to_parents_rots(t2m_kinematic_chain, rots)
    anim = Animation(new_rots, offs, Quaternions.id(offs.shape[0]), offs[0], parents) 
    return anim


def save_motion(anim, save_dir, clip_name, sample_description):
    plot_3d_motion(anim, pjoin(save_dir, clip_name), figsize=(10, 10), fps=12.5, radius=10, title = sample_description)


if __name__=="__main__":
    hml_vec = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/new_joint_vecs/000010.npy")
    hml_as_truebones_compact_rep = hml_rep_to_truebones_tensor(hml_vec, smpl_offsets, t2m_kinematic_parents)
    # augment with new func 
    anim1 = get_motion_inner(hml_as_truebones_compact_rep, t2m_kinematic_parents, len(t2m_kinematic_parents))
    plot_3d_motion(anim1, pjoin("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/test_output", "hml_fk.mp4"))

    # # augment with old func
    # hml_as_truebones_full_rep = compact_rep_to_full_rep(hml_as_truebones_compact_rep, t2m_kinematic_parents)
    # augmented2, new_parents2, permutations2 = spatial_augmentation_full_rep(hml_as_truebones_full_rep, t2m_kinematic_parents, permutation_ind=0)
    # compact_rep2, parents_ = full_rep_to_compact_rep(augmented2)
    # anim2 = get_motion_inner(compact_rep2, new_parents2, len(new_parents2))
    # plot_3d_motion(anim2, pjoin("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/test_output", "old_aug_func1.mp4"))
    # # hml_root_quat, hml_root_pose = recover_root_rot_pos(torch.from_numpy(hml_vec))
    # # tb_root_quat, tn_root_pose = recover_root_quat_and_pos_np(hml_as_truebones_compact_rep[:, 0])
    # # tb_root_quat
    