from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model_wo_clip
from utils import dist_util
# from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_mixed_collate
from data_loaders.truebones.utils.motion_process import recover_from_ric_np, recover_from_rot_np ,recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from data_loaders.humanml.common.skeleton import Skeleton
from sklearn.preprocessing import normalize
from model.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions
import pathlib

process_dir_path = "save/0_to_render"

npy_dir = [f_path for f_path in list(pathlib.Path(process_dir_path).rglob("*")) if f_path.suffix == '.npy']
cond = np.load("dataset/truebones/zoo/processed_bvh_rots/cond.npy", allow_pickle=True).item()
for npy_f in npy_dir:
    object_type = os.path.basename(npy_f).split('__')[0]
    motion = np.load(npy_f)
    parents=cond[object_type]["parents"]
    offsets=cond[object_type]["offsets"]
    joints_names=cond[object_type]["joints_names"]
    global_positions, _ = recover_from_bvh_ric_np(motion, parents)
    out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
    out_path = os.path.dirname(npy_f)
    pref = os.path.basename(npy_f)[:-4]
    plot_general_skeleton_3d_motion(pjoin(out_path, f'{pref}_fixed.mp4'), parents, global_positions, dataset="truebones", title="", fps=20)
    BVH.save(pjoin(out_path, f'{pref}_fixed.bvh'), out_anim, joints_names)