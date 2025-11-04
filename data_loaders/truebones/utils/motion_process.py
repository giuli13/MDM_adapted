import BVH
import typing
from Animation import *
from InverseKinematics import animation_from_positions
import numpy as np 
import os 
from os.path import join as pjoin
from Quaternions import Quaternions
from visualizations import plot_3d_motion
import re
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from data_loaders.humanml.utils.plot_script import plot_general_skeleton_3d_motion
import random
import math
import itertools
import statistics
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import smpl_offsets, t2m_kinematic_parents, joint_names, t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.scripts.motion_process import process_file as hml_process
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos as hml_recover_root_rot_pos

from utils.parents_traverse import depth_list
from sklearn.preprocessing import normalize
from data_loaders.humanml.common.skeleton import Skeleton
import bisect
from data_loaders.get_data import get_dataset_loader
import re 

##  Objects which are oriented upside-down (grow towards negative y-axis)
UPSIDE_DOWN_GROUP = ["Bear", "Dog", "Dog-2", "Roach", "SabreToothTiger", "Deer", "Scorpion", "Spider", "FireAnt"]
## objects which are oriented same as HML and therefore should not be rotated
NO_ROTATION_GROUP = ["Raptor2", "Raptor3", "Rat", "Pigeon", "Scorpion-2", "Pteranodon", "Trex", "Mousey_m", "MouseyNoFingers"]
DATA_DIR = "dataset/truebones/zoo/Truebone_Z-OO"
DATA_DIR_MIXAMO = "dataset/mixamo"
# TEST_OUTPUT_DIR = "dataset/truebones/zoo/test_output"
TEST_OUTPUT_DIR = "dataset/truebones/zoo/processed_test"
SAVE_DIR = "dataset/truebones/zoo/processed_bvh_rots"
AUGMENTED_SAVE_PREFIX = "dataset/truebones/zoo/processed/augmented"
HML_DATA_PATH= "dataset/HumanML3D"
HML_MOTION_DIR = "new_joint_vecs"
HML_TEXT_DIR = "texts"
TEXT_DIR = "texts"
MOTION_DIR = "motions"
ANIMATIONS_DIR = "animations"
JOINT_RELATIONS_DIR = "joint_relation"
JOINT_GRAPH_DIST_DIR = "graph_dist"
NORMALIZED_BONE_LENGTH_DIR = "bone_len_diff"
FOOT_CONTACT_HEIGHT_THRESH = 0.3
FOOT_CONTACT_VEL_THRESH = 0.002
MAX_PERMS = 25
AUGMENTATIONS_PER_MOTION = 10
MAX_PATH_LEN = 5.
MEAN_DIR = "mean"
STD_DIR = "std"
SPECIAL_CARE_OBJECT_TYPES = []
COSMETICS = ["PolarBearB", "KingCobra", "Hamster", "Skunk", "Comodoa", "Hippopotamus", "Leapord", "Rhino", "Hound"]
SNAKES = ["KingCobra", "Anaconda"]
NO_HANDS = ["Raptor"]
NO_BVHS =["Jaws", "Crow", "Dog", "Dog-2"]
IGNORE_OBJECTS = NO_BVHS #SPECIAL_CARE_OBJECT_TYPES + COSMETICS + SNAKES + NO_HANDS + NO_BVHS
TEST_OBJECTS =  SPECIAL_CARE_OBJECT_TYPES + COSMETICS + NO_HANDS
checkSizes = ["PolarBearB", "PolarBear", "Elephant", "Jaguar", "Goat", "Coyote", "Hippopotamus", "Mammoth", "BrownBear", "Gazelle"]
INSECTS = ["Cricket", "SpiderG" , "Scorpion", "Isopetra", "FireAnt", "Crab", "Centipede", "Roach", "Ant", "HermitCrab", "Scorpion-2", "Spider"]
SNAKES = ["Anaconda", "KingCobra"]
FLYING_OBJECTS = ["Bat", "Dragon", "Bird", "Buzzard", "Eagle", "Giantbee", "Parrot", "Parrot2", "Pigeon", "Pteranodon", "Tukan"]
CONNECTED_TO_GROUND = ["Bear", "Camel", "Hippopotamus", "Horse", "Pirrana", "Pteranodon", "Raptor3", "Rat", "SabreToothTiger", "Scorpion-2", "Spider", "Trex", "Tukan", "Pirrana"]
FISH = ["Pirrana"]
DINOSAURS = ["Ostrich", "Flamingo", "Raptor", "Raptor2", "Raptor3", "Trex", "Chicken", "Tyranno"]
MAMMALS = ["Horse", "Hippopotamus", "Comodoa", "Camel", "Bear", "Buffalo", "Cat", "BrownBear", "Coyote", "Crocodile", "Elephant", "Deer", "Fox", "Gazelle", 
           "Goat", "Jaguar","Lynx", "Tricera", "Stego" , "SandMouse", "Raindeer", "Puppy", "PolarBear", "Monkey", "Mammoth", "Alligator", "Hamster", 
           "Hound", "Leapord", "Lion", "PolarBearB", "Rat", "Rhino", "SabreToothTiger", "Skunk", "Turtle"]
OBJECT_SUBSETS_DICT = {
    "all" : MAMMALS + DINOSAURS + INSECTS + SNAKES + FISH + FLYING_OBJECTS,
    "mammals": MAMMALS,
    "mammals_no_sand_mouse": [mammal for mammal in MAMMALS if mammal != "SandMouse"],
    "mammals_no_cat": [mammal for mammal in MAMMALS if mammal != "Cat"],
    "mammals_no_comodoa": [mammal for mammal in MAMMALS if mammal != "Comodoa"],
    "connected_to_ground": CONNECTED_TO_GROUND,
    "flying": FLYING_OBJECTS,
    "dinosaurs": DINOSAURS,
    "insects": INSECTS,
    "insects_snakes": INSECTS + SNAKES,
    "insects_snakes_no_crab": [insect for insect in INSECTS + SNAKES if insect != "Crab"],
    "mammals_clean": [mammal for mammal in MAMMALS if mammal not in CONNECTED_TO_GROUND],
    "insects_clean": [insect for insect in INSECTS if insect not in CONNECTED_TO_GROUND],
    "dinosaurs_clean": [dinosaur for dinosaur in DINOSAURS if dinosaur not in CONNECTED_TO_GROUND],
    "flying_clean": [fly for fly in FLYING_OBJECTS if fly not in CONNECTED_TO_GROUND],
    "all_clean": [obj for obj in  MAMMALS + DINOSAURS + INSECTS + SNAKES + FISH + FLYING_OBJECTS if obj not in CONNECTED_TO_GROUND]
}
FACE_JOINTS = {"Alligator": [8, 11, 17, 20] , "Crow": [18, 21, 7, 11], "Anaconda": [13, 26, 13, 26], "Ant": [9, 15, 23, 30], "Bat": [6, 15, 26, 34], 
               "Bear": [8, 2, 39, 59], "Bird": [15, 35, 6, 11], "BrownBear": [2, 7, 15, 23], "Buffalo": [6, 12, 21, 27], "Buzzard": [7, 23, 41, 47], "Camel": [9, 15, 33, 27], "Cat": [6, 12, 22, 28], 
               "Centipede": [7, 2, 41, 47], "Chicken": [5, 17, 30, 32], "Comodoa": [11, 1, 34, 44], "Coyote": [5, 11, 21, 28], "Crab": [14, 20, 51, 47], 
               "Cricket": [20, 25, 32, 36], "Crocodile": [7, 12, 21, 27], "Deer": [4, 9, 30, 36], "Dog": [8, 14, 26, 32], "Dog-2": [8, 14, 26, 32], 
               "Dragon":[10, 23, 47, 83], "Eagle": [7, 20, 35, 41], "Elephant": [6, 10, 32, 36], "FireAnt": [15, 19, 25, 29], 
               "Flamingo": [15, 22, 10, 6], "Fox": [27, 33, 15, 8], "Gazelle": [4, 10, 20, 26], "Giantbee": [11, 16, 3, 1], "Goat": [24, 19, 14, 8], "Hamster": [3, 9, 19, 25], 
               "HermitCrab": [51, 46, 8, 12], "Hippopotamus": [5, 11, 29, 35], "Horse": [10, 16, 33, 41], "Hound": [3, 9, 19, 25], "Isopetra": [48, 55, 18, 26], "Jaguar": [6, 12, 22, 28], 
               "KingCobra": [6, 7, 6, 7], "Leapord": [7, 13, 26, 32], "Lion": [6, 11, 20, 25], "Lynx": [2, 8, 18, 24], "Mammoth": [7, 11, 34, 38], 
               "Monkey": [9, 21, 57, 37], "Ostrich": [6, 16, 36, 28], "Parrot": [9, 25, 65, 43], "Parrot2": [7, 23, 42, 48], "Pigeon": [3, 4, 1, 6], "Pirrana": [19, 20, 4, 5], 
               "PolarBear": [3, 9, 19, 25], "PolarBearB": [3, 8, 17, 23], "Pteranodon": [16, 5, 40, 35], "Puppy": [5, 11, 20, 26], "Raindeer": [3, 9, 18, 24], "Raptor": [13, 19, 13, 19], 
               "Raptor2": [52, 40, 23, 13], "Raptor3": [53, 41, 24, 14], "Rat": [12, 15, 9, 6], "Rhino": [5, 11, 22, 28], "Roach": [2, 6, 29, 25], "SabreToothTiger": [7, 2, 38, 52], 
               "SandMouse": [7, 13, 30, 34], "Scorpion": [58, 29, 20, 25], "Scorpion-2": [55, 23, 48, 16], "Skunk": [10, 15, 28, 32], "Spider": [21, 27, 5, 9], "SpiderG": [13, 19, 27, 33],
               "Stego": [7, 12, 27, 21], "Trex": [38, 50, 23, 15], "Tricera": [6, 11, 24, 28], "Tukan": [4, 6, 9, 11], "Turtle": [31, 40, 12, 22], "Tyranno": [7, 20, 37, 44], "Mousey_m": [55, 50, 29, 8], 
               "MouseyNoFingers": [23, 18, 13, 8], "Human": [2, 1, 17, 16]}

HML_AVG_BONELEN = statistics.mean(np.linalg.norm(smpl_offsets[1:], axis=1))

""" return qs rotatation based on object type """
def get_rotation_qs(object_type):
    if object_type in NO_ROTATION_GROUP:
        return Quaternions.from_euler(np.array([0, 0, 0]), "xyz")
    elif object_type in UPSIDE_DOWN_GROUP:
        return Quaternions.from_euler(np.array([0, np.pi, 0]), "xyz")
    return Quaternions.from_euler(np.array([0, -np.pi/2, 0]), "xyz")

""" return qs rotatation based on object type """
def get_rotation_qs_accurate(anim, object_type, face_joint_indx=None):
    global_positions = positions_global(anim)
    if face_joint_indx is None:
        face_joint_indx = FACE_JOINTS[object_type]
    root_pos_init = global_positions[0]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = Quaternions.between(forward_init, target)
    return root_quat_init

""" get local velocity of all joints """
def get_local_velocity(global_positions, rotations_quaternions):
    global_positions_ = global_positions.copy()
    rotations_quaternions_ = rotations_quaternions.copy()
    r_rot = rotations_quaternions_[:, 0] # (Frames, 4)
    velocity = global_positions_[1:] - global_positions_[:-1]
    r_rots = r_rot[:-1, None].repeat(velocity.shape[1], axis = 1) # (Frames - 1, 4)
    return r_rots * velocity # (Frames - 1, joints, 3)

""" get root xz linear velocity """
def get_root_xz_linear_velocity(global_positions, r_quats):
    velocity = (global_positions[1:, 0] - global_positions[:-1, 0]).copy()
    velocity = r_quats[1:] * velocity
    return velocity[:, [0,2]]

""" put skeleton on ground """
def put_on_ground(anim, ground_height=None):
    if ground_height is None:
        t_pos_global_positions = positions_global(anim)
        ground_height = t_pos_global_positions.min(axis=0).min(axis=0)[1]
    new_positions = anim.positions.copy()
    new_positions[:, 0, 1] -= ground_height
    new_offsets = anim.offsets.copy()
    new_offsets[0, 1] -= ground_height
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, ground_height

""" move motion s.t root xz are at origin on first frame"""
def move_xz_to_origin(anim, root_pose_init_xz=None):
    if root_pose_init_xz is None:
        root_pos_init = anim.positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_pose_init_xz
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_pose_init_xz
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_pose_init_xz

"""" rotate the motion to initially face z+, ground at xz axis (negative y is below ground)"""
def rotate_to_hml_orientation(anim, object_type):
    qs_rot = get_rotation_qs(object_type)
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_offs = anim.offsets.copy()
    new_offs[0] = qs_rot * new_offs[0]
    new_anim = Animation(new_rots,  new_pos, anim.orients.copy(), new_offs, anim.parents.copy())
    return new_anim

"""" rotate the motion to initially face z+, ground at xz axis (negative y is below ground)"""
def rotate_to_hml_orientation_accurate(anim, object_type, face_joints=None):
    qs_rot = get_rotation_qs_accurate(anim, object_type, face_joint_indx=face_joints)
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim


""" scale skeleton s.t longest armature is of length 1 """
def scale(anim, scale_factor=None):
    if scale_factor is None:
        lengths = offset_lengths(anim)
        mean_len = statistics.mean(lengths)
        scale_factor = HML_AVG_BONELEN/mean_len
    new_anim = Animation(anim.rotations.copy(), anim.positions * scale_factor ,anim.orients.copy(), anim.offsets * scale_factor,
                         anim.parents.copy())
    return new_anim, scale_factor

""" get foot contact """
def get_foot_contact(positions, foot_joints_indices, vel_thresh):
    frames_num, joints_num = positions.shape[:2]
    foot_vel_x = (positions[1:,foot_joints_indices ,0] - positions[:-1,foot_joints_indices ,0]) ** 2
    foot_vel_y = (positions[1:, foot_joints_indices, 1] - positions[:-1, foot_joints_indices, 1]) **2
    foot_vel_z = (positions[1:, foot_joints_indices, 2] - positions[:-1, foot_joints_indices, 2]) **2
    total_vel = foot_vel_x + foot_vel_y + foot_vel_z
    foot_contact_vel_map = np.where(np.logical_and(total_vel <= vel_thresh, np.abs(positions[1:, foot_joints_indices,1]) <= FOOT_CONTACT_HEIGHT_THRESH), 1, 0)
    foot_cont = np.zeros((frames_num-1, joints_num))
    foot_cont[:, foot_joints_indices] = foot_contact_vel_map.astype(int)

    return foot_cont

""" detect foot from tpos """
def detect_foot_joints(t_pose_positions, contact_thresh):
    t_pose_positions = t_pose_positions[0]
    return np.where(t_pose_positions[:,1] < contact_thresh)

""" get 6d rotations continuous representation"""
def get_6d_rep(qs):
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)

"""" process anim object """
def process_anim(anim, object_type, root_pose_init_xz=None, scale_factor=None, ground_height=None):
    rotated = rotate_to_hml_orientation(anim, object_type)
    centered, root_pose_init_xz_ = move_xz_to_origin(rotated, root_pose_init_xz)
    grounded, ground_height_ = put_on_ground(centered, ground_height)
    scaled, scale_factor_ = scale(grounded, scale_factor)
    return scaled, root_pose_init_xz_, ground_height_, scale_factor_


"""" process anim object """
def process_anim_accurate(anim, object_type, root_pose_init_xz=None, scale_factor=None, ground_height=None, face_joints=None):
    rotated = rotate_to_hml_orientation_accurate(anim, object_type, face_joints) 
    centered, root_pose_init_xz_ = move_xz_to_origin(rotated, root_pose_init_xz)
    scaled, scale_factor_ = scale(centered, scale_factor)
    grounded, ground_height_ = put_on_ground(scaled, ground_height)
    return grounded, root_pose_init_xz_, ground_height_, scale_factor_


def face_z_positive(anim, object_type):
    gpos = positions_global(anim)
    pos_init = gpos[0]
    face_joints = FACE_JOINTS[object_type]
    r_hip, l_hip, sdr_r, sdr_l = face_joints
    across1 = pos_init[r_hip] - pos_init[l_hip]
    across2 = pos_init[sdr_r] - pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = Quaternions.between(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = root_quat_init * positions
    

def get_valid_string(s, spaces=False):
    s_valid = re.sub("(^|[ _])\s*([a-zA-Z])", lambda p: p.group(0).upper(), s)
    s_valid = ''.join(filter(str.isalpha, s_valid))
    if spaces:
        words = re.findall('[A-Z][^A-Z]*', s_valid)
        s_valid = " ".join(words)
    return s_valid


def get_text(object_type, action):
    obj = get_valid_string(object_type, spaces=False)
    act = get_valid_string(action, spaces=True)
    caption =  "The " + obj + " " + act
    words = caption.split(' ')
    caption_words  = list()
    cur = ""
    for word in words:
        if word != cur:
            caption_words.append(word)
            cur = word
    return " ".join(caption_words)


def get_tokens(caption):
    tokens = caption.split(' ')
    tokens[0] = tokens[0]+"/DET"
    tokens[1] = tokens[1]+"/NOUN"
    tokens[2] = tokens[2]+"/VERB"
    for i in range(3, len(tokens)):
        tokens[i] = tokens[i]+"/ADJ"
    return tokens

def get_all_joints_in_subtree(parents, joints, root):
    joints.append(root)
    children = [j for j in range(len(parents)) if parents[j] == root]
    for child in children:
        get_all_joints_in_subtree(parents, joints, child)
        
def remove_subtree(anim, joint_lst, names):
    # removes all joints in the subtree, not including joint_lst
    remove_joints = list()
    for joint in joint_lst:
        get_all_joints_in_subtree(anim.parents, remove_joints, joint)
    included_joints = joint_lst + [i for i in range(len(anim.parents)) if i not in remove_joints]
    included_joints = sorted(included_joints)
    anim_rotations = anim.rotations[:, included_joints]
    anim_positions = anim.positions[:, included_joints]
    anim_offsets = anim.offsets[included_joints]
    anim_orients = anim.orients[included_joints]
    anim_parents = anim.parents[included_joints]
    new_names = [name for i, name in enumerate(names) if i in included_joints]
    reverse_map = [i for i in range(len(anim.parents))]
    for i in range(len(included_joints)):
        reverse_map[included_joints[i]] = i
    for j, p in enumerate(anim_parents[1:], 1):
        anim_parents[j] = reverse_map[p]
    return Animation(anim_rotations, anim_positions, anim_orients, anim_offsets, anim_parents), new_names
    
""" get object_type common characteristics, extracted from Tsode bvh"""
def get_common_features_from_T_pose(t_pose_bvh, object_type, foot_contact_height_thresh):
    t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pose_bvh)
    p1 = t_pose_anim.offsets[None, :].repeat(t_pose_anim.positions.shape[0], axis=0)
    p1[:, 0] = t_pose_anim.positions[:, 0]
    t_pose_anim = Animation(t_pose_anim.rotations, p1, t_pose_anim.orients, t_pose_anim.offsets, t_pose_anim.parents)
    ground_height=None
    if object_type == "Dragon":
        ground_height=0
    scaled, root_pose_init_xz, ground_height, scale_factor = process_anim(t_pose_anim, object_type, ground_height=ground_height)
    t_pos = positions_global(scaled)
    offsets = offsets_from_positions(t_pos, scaled.parents)[0]
    suspected_foot_indices = detect_foot_joints(t_pos, foot_contact_height_thresh)
    positions_for_anim = offsets.copy()[None,:].repeat(scaled.positions.shape[0], axis=0)
    positions_for_anim[:, 0] = offsets_global(scaled)[None, 0].repeat(scaled.positions.shape[0], axis=0)

    return root_pose_init_xz, scale_factor, ground_height, offsets, suspected_foot_indices[0], scaled.rotations, t_pos_names, scaled

""" get object_type common characteristics, extracted from Tsode bvh"""
def get_common_features_from_T_pose_accurate(t_pose_bvh, object_type, face_joints=None):
    t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pose_bvh)
    # first recover global positions, and then create a brand new non-damaged animation, with position consistent to the offsets 
    t_pose_positions = positions_global(t_pose_anim)

    t_pose_anim, _1, _2 = animation_from_positions(positions=t_pose_positions, parents=t_pose_anim.parents, offsets=t_pose_anim.offsets, iterations=150)
    ground_height=None
    if object_type == "Dragon":
        ground_height=0
    scaled, root_pose_init_xz, ground_height, scale_factor = process_anim_accurate(t_pose_anim, object_type, ground_height=ground_height, face_joints=face_joints)
    offsets = offsets_from_positions(positions_global(scaled), scaled.parents)[0]
    if object_type in ["Anaconda", "KingCobra"]: # special handel for snakes 
        suspected_foot_indices = [i for i in range(len(t_pos_names))]
    else:
        suspected_foot_indices = [i for i in range(len(t_pos_names)) if 'toe' in t_pos_names[i].lower() or 'foot' in t_pos_names[i].lower() or 
                                  'phalanx' in t_pos_names[i].lower() or 'hoof' in t_pos_names[i].lower() or 'ashi' in t_pos_names[i].lower()]
                # edge cases
        for si in suspected_foot_indices:
            if si in t_pose_anim.parents:
                #check if all childeren also in suspected_foot_indices, otherwise add them 
                children = [i for i in range(len(t_pos_names)) if t_pose_anim.parents[i] == si]
                for c in children:
                    if c not in suspected_foot_indices:
                        suspected_foot_indices.append(c)
    return root_pose_init_xz, scale_factor, ground_height, offsets, suspected_foot_indices, scaled.rotations, t_pos_names, scaled

""" motion features, shape (Frames, joints, joints, features_len) """
def get_motion_features(parents, ric_positions, rotations, foot_contact, velocity, offsets, max_joints, root_data):
    # F = Frames# , J = joints# 
    # parents (J,1)
    # positions (F, J, 3)
    # rotations (F, J, 6)
    # foot_contact (F - 1, J, 1)
    # velocity (F - 1, J, 3)
    # offsets (J, 3)
    
    # feature len = 13 (pos, rot, vel, foot)

    frames, joints = ric_positions.shape[0:2]
    if joints > max_joints:
        max_joints = joints
    features_len = 13
    features = np.zeros((frames - 1, joints, features_len))
    features[:, 0, :4] = root_data.copy()
    pos = ric_positions[:-1, 1:]  ## (Frames-1, joints-1, 3)
    # p_pos = ric_positions[:-1,parents[1:]]## (Frames-1, joints-1, 3)
    # for chain in kin_chains:
    #     if chain[0] == 0:
    #         continue
    #     else:
    #         p_pos[:, chain[0]-1] = ric_positions[:, 0]
    rot = rotations[:-1, 1:] ## (frames -1, joints - 1, 6)
    p_rot = rotations[:-1, parents[1:]] ## (frames -1, joints - 1, 6)
    vel = velocity[:, 1:] ## (Frames - 1, joints - 1, 3)
    p_vel = velocity[:,parents[1:]] ## (Frames - 1, joints - 1, 3)
    foot = foot_contact[:, 1:].reshape(frames - 1, joints - 1, 1) ## (Frames - 1, 1)
    normalized_offsets = normalize(offsets, axis=1)[1:]
    offsets_len = np.sum(offsets[1:] ** 2.0, axis=1) ** 0.5
    features[:, 1:] = np.concatenate([pos, rot, vel, foot], axis=-1) 
    return features, max_joints

'''return positions in root coords system. Meaning, each frame faces Z+, and the root is at [0, root_height, 0]'''
def get_rifke(global_positions, root_rot):
    positions = global_positions.copy()
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = np.repeat(root_rot[:, None], positions.shape[1], axis=1) * positions
    return positions


""" compute new rotations for anim which are compatible to multiply inverse rotations on the right """
def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots


def get_cont6d_params_old(anim):
    global_positions = positions_global(anim)
    # (seq_len, joints_num, 4)
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(anim.rotations)
    # (seq_len, 4)
    r_rot = anim.rotations[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (global_positions[1:, 0] - global_positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len - 1, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot, global_positions

def object_policy(obj):
    if obj in ["Mousey_m", "MouseyNoFingers", "Scorpion", "Raptor2"]:
        return "l_first"
    else:
        return "h_first"
    
def get_cont6d_params(anim, object_type, face_joints=None):
    parents = anim.parents
    positions = positions_global(anim)
    offsets = anim.offsets.copy()
    
    normalized_offsets = normalize(offsets, axis=1)
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    skel = Skeleton(torch.from_numpy(normalized_offsets), kinematic_chains, "cpu")
    if face_joints is None:
        face_joints = FACE_JOINTS[object_type]
    quat_params = Quaternions(skel.inverse_kinematics_np(positions, face_joints, smooth_forward=True))
    
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot, positions


def get_root_quat(joints, face_joint_idx, smooth_forward=False):
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        # across1 = across1 /  np.sqrt((across1**2).sum(axis=-1))[:, np.newaxis]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        # across2 = across2 /  np.sqrt((across2**2).sum(axis=-1))[:, np.newaxis]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = Quaternions.between(forward, target)
        return root_quat
        
        
def get_bvh_cont6d_params(anim, object_type, face_joints=None):
    positions = positions_global(anim)
    if face_joints is None:
        face_joints = FACE_JOINTS[object_type]
    
    quat_params = anim.rotations
    r_rot = get_root_quat(positions, face_joints, smooth_forward=False)
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(quat_params)
    cont_6d_params_reordered = np.zeros_like(cont_6d_params)
    for j, p in enumerate(anim.parents[1:], 1):
        cont_6d_params_reordered[:, j] = cont_6d_params[:, p]
    cont_6d_params_reordered[:, 0] = get_6d_rep(r_rot)

    
    # (seq_len, 4)
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len, joints_num, 4)
    return cont_6d_params_reordered, r_velocity, velocity, r_rot, positions

def get_hml_aligned_anim(bvh_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error):
    raw_anim, names, frame_time = BVH.load(bvh_path)
    correct_positions = raw_anim.offsets[None, :].repeat(raw_anim.positions.shape[0], axis=0)
    correct_positions[:, 0] = raw_anim.positions[:, 0].copy()
    anim = Animation(raw_anim.rotations, correct_positions, raw_anim.orients, raw_anim.offsets, raw_anim.parents)
    frames_num, joints_num = anim.positions.shape[:2]
    squared_positions_error[bvh_path] = np.sum((anim.positions - raw_anim.positions) ** 2)/(anim.positions.shape[0]*anim.positions.shape[1])
    print("positions mismatch error for file: " + bvh_path + " is " + str(squared_positions_error[bvh_path]))

    ## process animation: rotate to correct orientation, center, put on ground and scale
    processed_anim, _xz, _gh, _sf = process_anim(anim, object_type, root_pose_init_xz, scale_factor, ground_height)

    ## create new animation object in which the rotations are w.r.t the actual Tpos
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    ## create animation object which is defined over correct tpos
    new_anim = Animation(rots, anim_positions, processed_anim.orients, offsets, processed_anim.parents)
    return new_anim, names 

def get_hml_aligned_anim_accurate(bvh_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, face_joints=None):
    if not isinstance(bvh_path, Animation):
        raw_anim, names, frame_time = BVH.load(bvh_path)
        print('frame time', frame_time )
        frames_num, joints_num = raw_anim.positions.shape[:2]
        squared_positions_error[bvh_path] = 0 #np.sum((global_pos - new_global_pos) ** 2)/(anim.positions.shape[0]*anim.positions.shape[1])
        print("positions mismatch error for file: " + bvh_path + " is " + str(squared_positions_error[bvh_path]))

        ## process animation: rotate to correct orientation, center, put on ground and scale
        processed_anim, _xz, _gh, _sf = process_anim_accurate(raw_anim, object_type, root_pose_init_xz, scale_factor, ground_height, face_joints=face_joints)
    else:
        names = list()
        processed_anim = bvh_path
        frames_num = len(processed_anim)

    ## create new animation object in which the rotations are w.r.t the actual Tpos
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    # create animation object which is defined over correct tpos 
    new_anim = Animation(rots, anim_positions  , processed_anim.orients, offsets, processed_anim.parents)

    return new_anim, names 

""" plot single frame"""
def plot_single_frame(positions, frame_num, foot_contact, parents, save_dir, fig_name):
    fig = plt.figure(figsize=(7,7))
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    ax.view_init(0, -90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for joint, parent in enumerate(parents[1:], start=1):
        color = "green"
        zorder = 5
        if foot_contact is not None:
            if foot_contact[frame_num][joint] == 1 or foot_contact[frame_num][parent] == 1:
                color = 'red'
                zorder = 10
        ax.plot3D(positions[frame_num,[joint, parent], 0], positions[frame_num, [joint, parent], 1],
                positions[frame_num, [joint, parent], 2], color=color, linewidth=2, linestyle='-', marker='o', zorder=zorder)
    plt.savefig(pjoin(save_dir, fig_name))

    

""" plot single frame kinchains"""
def plot_single_frame_kinchains(positions, frame_num, kinchains, save_dir, fig_name, names, mark_joints=[]):
    fig = plt.figure(figsize=(10,10))
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    ax.view_init(90, -90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    cmap = plt.get_cmap('tab20c')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kinchains))]
    
    for ind, chain in enumerate(kinchains):
        cor_names = [names[j] for j in chain]
        ax.plot3D(positions[frame_num, chain, 0], positions[frame_num, chain, 1],
            positions[frame_num, chain, 2], linewidth=2, linestyle='-', marker='o', color=colors[ind], label=cor_names[-1])
        if len(mark_joints) != 0:
            ax.scatter(positions[frame_num, mark_joints, 0], positions[frame_num, mark_joints, 1],
            positions[frame_num, mark_joints, 2], linewidth=2, marker='o', color='red')
    plt.legend(loc='best')
    plt.savefig(pjoin(save_dir, fig_name))      
        

""" get motion feature representation"""
def get_motion(bvh_path, foot_contact_vel_thresh, object_type, max_joints,root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=None):
    try:
        new_anim, names = get_hml_aligned_anim_accurate(bvh_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, face_joints)
        ## extract features
        cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(new_anim, object_type)
        foot_contact = get_foot_contact(global_positions, foot_indices, foot_contact_vel_thresh) 
        '''Get Joint Rotation Invariant Position Represention'''
        # local velocity wrt root coords system as described in get_rifke definition 
        positions = get_rifke(global_positions, r_rot)
        root_y = positions[:, 0, 1:2]
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        local_vel = np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
        features, max_joints = get_motion_features(new_anim.parents, positions, cont_6d_params, foot_contact, local_vel, offsets, max_joints, root_data) #, new_anim
        return features, new_anim.parents, max_joints, new_anim
    except Exception as err:
        print(err)
        return None, None, max_joints, None
    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = SAVE_DIR, face_joints=None):
    object_cond = dict()
    bvhs_dir = DATA_DIR
    if object_type in ["Mousey_m", "MouseyNoFingers"]:
        bvhs_dir = DATA_DIR_MIXAMO
    bvh_files = [pjoin(bvhs_dir, object_type, f) for f in os.listdir(pjoin(bvhs_dir, object_type)) if f.lower().endswith('.bvh')]
    if len(bvh_files) == 0:
        return files_counter, frames_counter, max_joints
    ## get t-pos bvh
    t_pos_path = None
    for f in bvh_files:
        if "tpos" in f.lower():
            t_pos_path = f
            break
    if t_pos_path is not None:
        bvh_files.remove(t_pos_path)
    else: #choose some other motion to be treated as tpos 
        for f in bvh_files:
            fnam = os.path.basename(f)
            if fnam.lower().startswith('idle') or fnam.lower().startswith('__idle'):
                t_pos_path = f
                break
    if t_pos_path is None:
        t_pos_path = bvh_files[0]

    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose_accurate(t_pos_path, object_type, face_joints=face_joints)
    t_pos_motion, parents, max_joints, new_anim = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=face_joints)
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    edge_len = np.array(offset_lengths(tpos_anim))
    normalized_edge_len = np.array(offset_lengths(tpos_anim) - HML_AVG_BONELEN) / np.std(edge_len)
    object_cond['normalized_edge_len'] = normalized_edge_len
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    object_cond['kinematic_chains'] = kinematic_chains
    
    all_tensors = list()
    
    for f in bvh_files:
        print("processing file: " + f)
        motion, parents, max_joints, new_anim = get_motion(f, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error)
        if motion is not None:
            _, file_name = os.path.split(f)
            action = file_name.split('.')[0]
            caption = get_text(object_type, action)
            tokens = get_tokens(caption)
            while motion.shape[0] >= 240:
                cur_motion = motion[:200]
                motion = motion[200:]
                cur_anim = Animation(rotations=new_anim.rotations[:200], positions=new_anim.positions[:200], parents=new_anim.parents, orients=new_anim.orients[:200], offsets=new_anim.offsets)
                new_anim = Animation(rotations=new_anim.rotations[200:], positions=new_anim.positions[200:], parents=new_anim.parents, orients=new_anim.orients[200:], offsets=new_anim.offsets)
                all_tensors.append(cur_motion)
                files_counter += 1
                frames_counter += cur_motion.shape[0]
                name = object_type + "_" + action + "_" + str(files_counter)
                np.save(pjoin(save_dir, MOTION_DIR, name + '.npy'), cur_motion)
                BVH.save(pjoin(save_dir, "bvhs", name+".bvh"), cur_anim, names)
                text_file = open(pjoin(save_dir, TEXT_DIR, name+'.txt'), "w")
                n = text_file.write(caption+ "#" + " ".join(tokens) + "#0.0#0.0#" + object_type + "#" + str(parents.tolist()) + "#"+ str(names))
                text_file.close()
                # create mp4 from rotations (sanity check)
                positions = recover_from_bvh_rot_np(cur_motion, parents, offsets)
                fc = [[j for j in range(len(parents)) if cur_motion[f, j , 12] != 0] for f in range(cur_motion.shape[0])]
                plot_general_skeleton_3d_motion(pjoin(save_dir, ANIMATIONS_DIR, name+".mp4"), parents, positions, dataset="truebones", title="", fps=20, face_joints=FACE_JOINTS[object_type], fc = fc)
            
            if motion.shape[0] > 0:
                all_tensors.append(motion)
                files_counter += 1
                frames_counter += motion.shape[0]
                name = object_type + "_" + action + "_" + str(files_counter)
                np.save(pjoin(save_dir, MOTION_DIR, name + '.npy'), motion)
                BVH.save(pjoin(save_dir, "bvhs", name+".bvh"), new_anim, names)
                text_file = open(pjoin(save_dir, TEXT_DIR, name+'.txt'), "w")
                n = text_file.write(caption+ "#" + " ".join(tokens) + "#0.0#0.0#" + object_type + "#" + str(parents.tolist()) + "#"+ str(names))
                text_file.close()
                # create mp4 from rotations (sanity check)
                positions = recover_from_bvh_rot_np(motion, parents, offsets)
                fc = [[j for j in range(len(parents)) if motion[f, j , 12] != 0] for f in range(motion.shape[0])]
                plot_general_skeleton_3d_motion(pjoin(save_dir, ANIMATIONS_DIR, name+".mp4"), parents, positions, dataset="truebones", title="", fps=20, face_joints=FACE_JOINTS[object_type], fc = fc)
                
                
                    
        else:
            print("failed to process file: "+f)
    all_tensors = np.concatenate(all_tensors, axis=0)
    mean, std = get_mean_std(all_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std

    return files_counter, frames_counter, max_joints, object_cond

""" extract parents and minimal features representation"""
def extract_minimal_features_and_parents(features):
    parents_map = features[0].sum(axis=2)
    dummy_joints = np.where(~parents_map.any(axis=1))[0]
    if dummy_joints.shape[0] == 1 and dummy_joints[0] == 0:
        dummy_joints = np.array([], dtype=int)
    mask = np.ones(features.shape, dtype=bool)
    mask[:, dummy_joints, dummy_joints, :] = False
    new_features = np.delete(features, dummy_joints, axis = 1)
    parents = [-1]
    for j in range(1, new_features.shape[1]):
        j_parent = np.where(parents_map[j] != 0)[0][0]
        parents.append(j_parent)
    return new_features, parents

def process_new_object_type(bvhs_dir, object_type,  sample_bvh, face_joints, save_dir=None, tpos_bvh=None):
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose_accurate(tpos_bvh, object_type, face_joints)
    t_pos_motion, parents, max_joints = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, None, face_joints=face_joints)
    object_cond = dict()
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    edge_len = np.array(offset_lengths(tpos_anim))
    normalized_edge_len = np.array(offset_lengths(tpos_anim) - HML_AVG_BONELEN) / np.std(edge_len)
    object_cond['normalized_edge_len'] = normalized_edge_len
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    object_cond['kinematic_chains'] = kinematic_chains
    all_tensors = list()
    all_bvhs = [pjoin(bvhs_dir, f) for f in os.listdir(bvhs_dir) if f.endswith('.bvh')]
    for bvh in all_bvhs:
        motion, parents, max_joints = get_motion(bvh, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, None)
        if motion is not None:
            all_tensors.append(motion)
            if bvh == sample_bvh:
                sample_motion = motion
    mean, std = get_mean_std(all_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std
    return sample_motion, object_cond

        
                

""" full rep (frames, joints, joints, feats) to compact rep (frames, joints, feats)"""
def full_rep_to_compact_rep(full_rep):
    motion, parents = extract_minimal_features_and_parents(full_rep)
    compact_rep = motion.sum(axis=-2) # (frames, max_joints, feature_len)
    return compact_rep, parents

""" compact rep (frames, joints, feats) to full rep (frames, joints, joints, feats) """
def compact_rep_to_full_rep(compact_rep, parents, ones = False):
    frames, joints, feats = compact_rep.shape
    if ones:
        full_rep = np.ones((frames, joints, joints, feats))    
    else:
        full_rep = np.zeros((frames, joints, joints, feats))
    full_rep[:, 0, 0] = compact_rep[:, 0]
    for j, p in enumerate(parents[1:], start=1):
        full_rep[:, j, p] = compact_rep[:, j]
    return full_rep

def kinematic_chains_to_parents(kinematic_chains):
    n_joints = max(max(kinematic_chains)) + 1
    parents = [0 for i in range(n_joints)]
    parents[0] = -1
    for chain in kinematic_chains:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j-1]
    return parents

def kinematic_chains_rots_to_parents_rots(kinematic_chains, rots):
    # rots Quaternions(n_frames, n_joints)
    new_rots = Quaternions.copy(rots)
    root_rots = new_rots[:, 0]
    cumulative_rots = Quaternions.copy(rots)
    
    for chain in kinematic_chains:
        prev_rot = root_rots
        for j in range(1, len(chain)):
            actual_rot = prev_rot * rots[:, chain[j]]
            relative_rot = -cumulative_rots[:, chain[j - 1]] * actual_rot
            new_rots[:, chain[j]] = relative_rot
            cumulative_rots[:, chain[j]] = cumulative_rots[:, chain[j-1]] * relative_rot
            prev_rot = actual_rot
    return new_rots
            


""" recover quaternions and positions from features for numpy only"""
def recover_root_quat_and_pos_np(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)

    r_rot_qs = np.zeros(data.shape[:-1] + (4,))
    r_rot_qs[... ,0] = np.cos(r_rot_ang)
    r_rot_qs[... ,2] = np.sin(r_rot_ang)

    r_rot_quat = Quaternions(r_rot_qs)

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
    r_pos[...,1] = data[..., 3]
    return r_rot_quat, r_pos

""" recover quaternions and positions from features for numpy only"""
def recover_root_quat_and_pos(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_rot_quat = Quaternions(r_rot_quat)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

""" recover xyz positions from ric (root relative positions) numpy """
def recover_from_ric_np(data, parents = None):
    if parents is None: # data must be in full representation 
        compact_rep, parents = full_rep_to_compact_rep(data)# (frames, max_joints, feature_len)
    else:
        compact_rep = data
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(compact_rep[:, 0])
    relative_positions = compact_rep[:, 1:, :3]
    '''Add Y-axis rotation to local joints'''
    positions = np.repeat(-r_rot_quat[:, None], relative_positions.shape[1], axis=1) * relative_positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions, parents
    

""" recover xyz positions from ric (root relative positions) torch """
def recover_from_bvh_ric_np(data, parents):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
    positions = data[..., 1:, :3]
    '''Add Y-axis rotation to local joints'''
    if len(r_rot_quat.shape) == 1:
        expanded_r_rot_quat = r_rot_quat[:, None]
    else:
        assert len(r_rot_quat.shape) == 2
        expanded_r_rot_quat = r_rot_quat[:, :, None]
        
    #positions = np.repeat(-expanded_r_rot_quat, relative_positions.shape[-2], axis=-2) * relative_positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions, parents
    

def recover_from_rot_np(data, skeleton):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:,0])

    r_rot_cont6d = get_6d_rep(r_rot_quat)

    start_indx = 3
    end_indx = 9
    cont6d_params = data[..., 1:, start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = np.concatenate([r_rot_cont6d[:, None, :], cont6d_params], axis=-2)
    positions = skeleton.forward_kinematics_cont6d(torch.from_numpy(cont6d_params).float(), torch.from_numpy(r_pos).float())

    return positions.detach().cpu().numpy()

def recover_from_bvh_rot_np(data, parents, offsets):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:,0])
    r_rot_cont6d = get_6d_rep(r_rot_quat)
    start_indx = 3
    end_indx = 9
    cont6d_params = data[..., 1:, start_indx:end_indx]
    cont6d_params = np.concatenate([r_rot_cont6d[:, None, :], cont6d_params], axis=-2)
    cont6d_params_hml_order = cont6d_to_matrix(cont6d_params)
    cont6d_params = np.eye(3)[None, None].repeat(cont6d_params.shape[0], axis=0).repeat(cont6d_params.shape[1], axis=1)
    for j, p in enumerate(parents[1:], 1):
        cont6d_params[:, p] = cont6d_params_hml_order[:, j]
    rotations = Quaternions.from_transforms(cont6d_params)
    positions = offsets[None].repeat(data.shape[0], axis=0)
    positions[:, 0] = r_pos
    anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))
    return positions_global(anim), anim

def recover_foot_contact(motion):
    _motion, parents = extract_minimal_features_and_parents(motion)
    foot_contact = np.zeros((motion.shape[0], motion.shape[1]))
    for j, p in enumerate(parents[1:], start=1):
        foot_contact[:, j] = motion[:, j, p, 15]
    return foot_contact 


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

def cont6d_to_matrix_torch(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    epsilon=1e-8
    cont6d = torch.nan_to_num(cont6d) + epsilon 
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    
    #x_norm = np.linalg.norm(x_raw, axis=-1, keepdims=True)
    #x = np.divide(x_raw, x_norm, out=np.zeros_like(x_raw), where=x_norm!=0)
    x = x_raw / torch.linalg.norm(x_raw, dim=-1, keepdims=True)
    z = torch.cross(x, y_raw, dim=-1)
    #z_norm = np.linalg.norm(z, axis=-1, keepdims=True) 
    z = z / torch.linalg.norm(z, dim=-1, keepdims=True)
    #z = np.divide(z, z_norm, out=np.zeros_like(z), where=x_norm!=0)
    
    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat

""" recover quaternions and offsets from data, numpy """
def recover_quats_and_offs_np(data, actual_joints_num, parents = None):
    if parents is None: # data must be in full representation 
        compact_rep, parents = full_rep_to_compact_rep(data)# (frames, max_joints, feature_len)
    else:
        compact_rep = data
    frames_num = compact_rep.shape[0]
    root_feature_vec = compact_rep[:, 0]
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(root_feature_vec)
    r_rot_cont6d = get_6d_rep(r_rot_quat)

    cont6d_params = np.zeros((frames_num, actual_joints_num, 6))
    offsets_params = np.zeros((frames_num, actual_joints_num, 3))

    rot_start_indx = 3
    rot_end_indx = rot_start_indx + 6
    off_start_indx = 12
    off_end_indx = off_start_indx + 3
        
    cont6d_params[:, 1:] = compact_rep[:, 1:actual_joints_num, rot_start_indx:rot_end_indx]
    offsets_params[:, 1:] = compact_rep[:, 1:actual_joints_num, off_start_indx:off_end_indx]

    cont6d_params[:,0] = r_rot_cont6d
    cont6d_params = cont6d_to_matrix(cont6d_params)
    offsets_params[:, 0] = r_pos
    return Quaternions.from_transforms(cont6d_params), offsets_params, parents

""" recover quaternions and offsets from data, torch """
def recover_quats_and_offs(data, actual_joints_num, parents = None):
    np_data = data.numpy()
    if parents is None:
        np_parents = None
    else:
        np_parents = parents.numpy()
    np_quats, np_offs, np_parents = recover_quats_and_offs_np(np_data, actual_joints_num, np_parents)
    quats = torch.from_numpy(np_quats, device=data.device)
    offs = torch.from_numpy(np_offs, device=data.device)
    parents = torch.from_numpy(np_parents, device = data.device)
    return quats, offs, parents 


def create_data_samples():
    ## prepare
    os.makedirs(pjoin(SAVE_DIR, TEXT_DIR), exist_ok=True)
    os.makedirs(pjoin(SAVE_DIR, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(SAVE_DIR, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(SAVE_DIR, "bvhs"), exist_ok=True)
    
    ## process
    # objects = [ d for d in os.listdir(DATA_DIR) if os.path.isdir(pjoin(DATA_DIR, d))]
    objects = [obj for obj in FACE_JOINTS.keys() if FACE_JOINTS[obj] != []]
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    
    for object_type in objects:
        if object_type in NO_BVHS or object_type == "Human":
            continue
        cur_counter = files_counter
        files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error)
        cond[object_type] = object_cond
        objects_counter[object_type] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(SAVE_DIR, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(SAVE_DIR, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(SAVE_DIR, "cond.npy"), cond)
    

def temporal_augmentation(motion, min_motion_length):
    frames_num = motion.shape[0]
    start_frame = random.randint(0, frames_num - min_motion_length)
    end_frame = random.randint(start_frame + min_motion_length, frames_num)
    augmented_motion = motion[start_frame:end_frame]
    return augmented_motion

def perm_given_index(alist, apermindex):
    alist = alist[:]
    for i in range(len(alist)-1):
        apermindex, j = divmod(apermindex, len(alist)-i)
        alist[i], alist[i+j] = alist[i+j], alist[i]
    return alist

def get_permutations(parents):
    n = len(parents)
    # get joint index with maximal # of children
    ranks = [0 for i in range(n)]
    for j, p in enumerate(parents[1:], 1):
        ranks[p] +=1
    max_rank_joint = np.argmax(ranks)
    chosen_joint_children = [j for j in range(n) if parents[j] == max_rank_joint]
    num_of_perms = np.math.factorial(len(chosen_joint_children))
    if num_of_perms > MAX_PERMS:
        all_permutations = list()
        step_size = num_of_perms // MAX_PERMS
        for i in range(0, num_of_perms, step_size):
            all_permutations.append(perm_given_index(chosen_joint_children, i))
    else:
        all_permutations = list(itertools.permutations(chosen_joint_children)) 
    return all_permutations, max_rank_joint

def parents_augmentation_full(parents):
    n = len(parents)
    num_of_perms = np.math.factorial(len(parents)-1)
    perm_ind = random.randint(0, num_of_perms - 1)
    permutation = [0] + perm_given_index([i+1 for i in range(n-1)], perm_ind)
    inv_perm = [0 for i in range(n)]
    for i, j in enumerate(permutation):
        inv_perm[j]=i
    new_parents = [-1] + [inv_perm[parents[permutation[i]]] for i in range(1, n)]
    return new_parents, permutation, num_of_perms

def parents_augmentation(parents):
    new_parents = np.array(parents)
    ## get permutation of joint with max rank
    all_permutations, permuted_joint = get_permutations(parents)
    perms_num = len(all_permutations)
    permotation_ind = random.randint(0, perms_num - 1)
    chosen_permutation = all_permutations[permotation_ind]
    orig_perm = all_permutations[0]
    joints_perm = [j for j in range(len(new_parents))]
    ## switch root children according to chosen permutations
    for i in range(len(orig_perm)):
        joints_perm[orig_perm[i]] = chosen_permutation[i]
        orig_children = [j for j in range(len(parents)) if parents[j] == chosen_permutation[i]]
        new_parents[orig_children] = orig_perm[i]

    ## propagate large values to bottom to maintain valid topology
    for subtree_root in orig_perm:
        children = [j for j in range(len(new_parents)) if new_parents[j] == subtree_root]
        while len(children) > 0:
            min_child = min(children)
            if min_child < subtree_root:
                # change min_child's children parent to subtree root
                min_child_children = [j for j in range(len(new_parents)) if new_parents[j] == min_child]
                if len(min_child_children) != 0:
                    new_parents[min_child_children] = subtree_root
                
                # change subtree_root (except for min_child) children's parent min_child
                for child in children:
                    if child != min_child:
                        new_parents[child] = min_child
                
                # switch min_child parent with subtree_root parent and set subtree_root parent to min_child
                new_parents[min_child] = new_parents[subtree_root]
                new_parents[subtree_root] = min_child

                # update joints perm
                joints_perm[min_child] = joints_perm[subtree_root]
                joints_perm[subtree_root] = min_child

                # recursive call 
                children = [j for j in range(len(new_parents)) if new_parents[j] == subtree_root]
            else:
                break
    return new_parents.tolist(), joints_perm, len(all_permutations)


def spatial_augmentation(motion, orig_parents):
    new_parents, joints_perm, permutations = parents_augmentation(orig_parents)
    augmented_motion = motion[:, joints_perm]
    return augmented_motion, new_parents, joints_perm, permutations

def spatial_augmentation_hard(motion, orig_parents):
    new_parents, joints_perm, permutations = parents_augmentation_full(orig_parents)
    augmented_motion = motion[:, joints_perm]
    return augmented_motion, new_parents, joints_perm, permutations

def spatial_augmentation_full_rep(motion, orig_parents):
    new_parents,joints_perm, permutations = parents_augmentation(orig_parents)
    if permutations == 0:
        ## could not apply this augmentation
        return motion, orig_parents, 0
    ## apply augmentation on motion
    augmented_motion = np.zeros_like(motion)
    augmented_motion[:, 0, 0, :] = motion[:, 0, 0, :]
    for j, p in enumerate(new_parents[1:], start=1):
        old_joint = joints_perm[j]
        old_parent = orig_parents[old_joint]
        augmented_motion[:, j, p, :] = motion[:, old_joint, old_parent, :]
    return augmented_motion, new_parents,  permutations

def augmentation_text(original_text, parents, joints_perm):
    parts = original_text.split('#')
    caption = parts[0]
    tokens = parts[1]
    tag_start = parts[2]
    tag_end = parts[3]
    object_type = parts[4]
    names = [e[1:-1] for e in parts[5].strip('][').split(', ')]
    names = [names[i] for i in joints_perm]
    if not isinstance(joints_perm, typing.List):
        joints_perm = joints_perm.tolist()
    return "#".join([caption, tokens, tag_start, tag_end, object_type, str(names), str(parents), str(joints_perm)])

def augmentation_text_hml(original_text, parents, names, joints_perm):
    parts = original_text.split('#')
    caption = parts[0]
    tokens = parts[1]
    tag_start = parts[2]
    tag_end = parts[3]
    object_type = "Human"
    names = [names[i] for i in joints_perm]
    if not isinstance(joints_perm, typing.List):
        joints_perm = joints_perm.tolist()
    return "#".join([caption, tokens, tag_start, tag_end, object_type, str(names), str(parents), str(joints_perm)])


def motion_augmentation(files_counter, frames_counter, original_motion, original_text, min_motion_length, num_of_augmentations, parents=None, temporal=True, spatial=True, save_prefix = AUGMENTED_SAVE_PREFIX):
    back_to_full_rep = parents is None
    if parents is None: # data must be in full representation 
        compact_rep, parents = full_rep_to_compact_rep(original_motion)# (frames, max_joints, feature_len)
    else:
        compact_rep = original_motion
    # save original motion-text pair 
    name = "0" * (6 - len(str(files_counter))) + str(files_counter)
    if back_to_full_rep:
        np.save(pjoin(save_prefix+"_motions", name + '.npy'), original_motion)
    text_file = open(pjoin(save_prefix+"_texts", name+'.txt'), "w")
    orig_joints_perm = np.arange(len(parents)).astype(int)
    new_text = augmentation_text(original_text, parents, orig_joints_perm)
    n = text_file.write(new_text)
    text_file.close()

    # increase counters
    files_counter+=1
    frames_counter+=compact_rep.shape[0]
    perm_ind = 1
    for i in range(num_of_augmentations):
        augmented_motion = compact_rep.copy()
        augmented_parents = parents.copy()
        joints_perm = orig_joints_perm 
        frames_num = augmented_motion.shape[0]

        if temporal and frames_num > min_motion_length:
            augmented_motion = temporal_augmentation(augmented_motion, min_motion_length)
        else:
            temporal=False

        if spatial:
            augmented_motion, augmented_parents, joints_perm, permutations = spatial_augmentation(augmented_motion, augmented_parents)
            if permutations == 1: # no joints with rank > 1, very unlikely
                spatial = False

        if temporal or spatial:
            # save augmentation
            name = "0" * (6 - len(str(files_counter))) + str(files_counter)
            if back_to_full_rep:
                full_rep_aug = compact_rep_to_full_rep(augmented_motion, augmented_parents)
            np.save(pjoin(AUGMENTED_SAVE_PREFIX+"_motions", name + '.npy'), full_rep_aug)
            text_file = open(pjoin(AUGMENTED_SAVE_PREFIX+"_texts", name+'.txt'), "w")
            new_text = augmentation_text(original_text, augmented_parents, joints_perm)
            n = text_file.write(new_text)
            text_file.close()
            # increase counters
            files_counter+=1
            frames_counter+=augmented_motion.shape[0]
            # check if there are any spatial augmentations left
            if spatial and perm_ind == permutations: #no more spatial permutations
                spatial=False

        else: # do not resume, there are no augmentation to apply 
            return files_counter, frames_counter
    return files_counter, frames_counter


def augment_dataset():
    ## prepare
    os.makedirs(AUGMENTED_SAVE_PREFIX + "_motions", exist_ok=True)
    os.makedirs(AUGMENTED_SAVE_PREFIX + "_texts", exist_ok=True)

    all_motions_npy = [f for f in os.listdir(pjoin(SAVE_DIR, MOTION_DIR))]
    files_counter = 0
    frames_counter = 0

    for motion_npy in all_motions_npy:
        print("Augmenting file: " + motion_npy.split('.')[0])
        motion = np.load(pjoin(SAVE_DIR, MOTION_DIR, motion_npy))
        text_file = open(pjoin(SAVE_DIR, TEXT_DIR, motion_npy.split('.')[0] + '.txt'), 'r')
        text = text_file.readlines()[0] 
        text_file.close()
        prev_files_counter = files_counter
        files_counter, frames_counter = motion_augmentation(files_counter, frames_counter, motion, text, 40, AUGMENTATIONS_PER_MOTION)
        print('Created %d augmented versions.' %(files_counter - prev_files_counter))
    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))

def object_mean_variance(object_type, test_ids):
    # for mousey
    partial_obj_type = object_type.split('_')[0]
    object_files = [file_name for file_name in os.listdir(pjoin(SAVE_DIR, MOTION_DIR)) if file_name.split('_')[0] == partial_obj_type and file_name.endswith('.npy')]
    train_data_list = list()
    test_data_list = list()
    for file in object_files:
        data = np.load(pjoin(SAVE_DIR, MOTION_DIR, file))
        if np.isnan(data).any():
            print(file)
            continue
        if file not in test_ids:
            train_data_list.append(data)
        else:
            test_data_list.append(data)
    save_mean_std(train_data_list, object_type, train=True)
    save_mean_std(test_data_list, object_type, train=False)

def truebones_split_files():
    test_ids = []
    train_ids = []
    object_types_list = [obj for obj in FACE_JOINTS.keys() if FACE_JOINTS[obj] != [] 
                         and obj not in SPECIAL_CARE_OBJECT_TYPES and obj != 'Human']
    for obj in object_types_list:
        partial_obj_type = obj.split('_')[0]
        object_files = [file_name for file_name in os.listdir(pjoin(SAVE_DIR, MOTION_DIR)) if file_name.split('_')[0] == partial_obj_type and file_name.endswith('.npy')]
        files_count = len(object_files)
        test_size = math.ceil(0.05 * float(files_count))
        random.shuffle(object_files)
        test_samples = random.sample(object_files, test_size)
        test_ids += test_samples
        for f in object_files:
            if f not in test_ids:
                train_ids.append(f)
                
    with open('test.txt', 'w') as f:
        for line in test_ids:
            f.write(f"{line}\n")
    with open('train.txt', 'w') as f:
        for line in train_ids:
            f.write(f"{line}\n")
    return test_ids, train_ids

def mean_variance(test_ids):
    #assuming compact representation
    os.makedirs(pjoin(SAVE_DIR, MEAN_DIR), exist_ok=True)
    os.makedirs(pjoin(SAVE_DIR, STD_DIR), exist_ok=True)
    object_types_list = [obj for obj in FACE_JOINTS.keys() if FACE_JOINTS[obj] != [] 
                         and obj not in SPECIAL_CARE_OBJECT_TYPES and obj != 'Human']
    for obj in object_types_list:
        object_mean_variance(obj, test_ids)
    

def get_mean_std(all_tensors):
    Mean = all_tensors.mean(axis=0) # (Joints, 13)
    Std = all_tensors.std(axis=0) # # (Joints, 13)
    
    Std[0, 4:] = 1.0 # root irrelevant cells
    Std[0, 0:1] = Std[0, 0:1].mean() / 1.0 # root rot vel
    Std[0, 1:3] = Std[0, 1:3].mean() / 1.0 # root x,z linear vel
    Std[0, 3:4] = Std[0, 3:4].mean() / 1.0 # root height

    
    Std[1:, :3] = Std[1:, :3].mean() / 1.0 # all joints except root ric pos
    Std[1:, 3:9] = Std[1:, 3:9].mean() / 1.0 # all joints except root rotation
    Std[1:, 9:12] = Std[1:, 9:12].mean() / 1.0 # all joints except root local velocity
    if len(Std[1:, 12][Std[1:, 12]!=0]) > 0:
            Std[1:, 12][Std[1:, 12]!=0] = Std[1:, 12][Std[1:, 12]!=0].mean() / 1.0 
    Std[1:, 12][Std[1:, 12]==0] = 1.0 # replace zeros with ones
    
    return Mean, Std

    
    
def save_mean_std(tensors_list, object_name, train=True):
    if len(tensors_list) > 0:
        data = np.concatenate(tensors_list, axis=0)
        Mean = data.mean(axis=0) # (Joints, 25)
        Std = data.std(axis=0) # # (Joints, 25)
        Std[0, 7:] = 1.0 # root irrelevant cells

        Std[0, 0:1] = Std[0, 0:1].mean() / 1.0 # root rot vel
        Std[0, 1:3] = Std[0, 1:3].mean() / 1.0 # root x,z linear vel
        Std[0, 3:4] = Std[0, 3:4].mean() / 1.0 # root height
        Std[0, 4:7] = Std[0, 4:7].mean() / 1.0 # root local velocity

        Std[1:, :3] = Std[1:, :3].mean() / 1.0 # all joints except root ric pos
        Std[1:, 3:9] = Std[1:, 3:9].mean() / 1.0 # all joints except root rotation
        Std[1:, 9:12] = Std[1:, 9:12].mean() / 1.0 # all joints except root local velocity
        if len(Std[1:, 12][Std[1:, 12]!=0]) > 0:
            Std[1:, 12][Std[1:, 12]!=0] = Std[1:, 12][Std[1:, 12]!=0].mean() / 1.0 
        Std[1:, 12][Std[1:, 12]==0] = 1.0 # replace zeros with ones
        
        ## same std for parents concat
        Std[1:, 13:16] = Std[1:, :3] # all joints except root ric pos
        Std[1:, 16:22] = Std[1:, 3:9] # all joints except root rotation
        Std[1:, 22:25] = Std[1:, 9:12] # all joints except root local velocity

        ## save mean, std files
        if train:
            np.save(pjoin(SAVE_DIR, MEAN_DIR,  f'{object_name}_Mean.npy'), Mean)
            np.save(pjoin(SAVE_DIR, STD_DIR,  f'{object_name}_Std.npy'), Std)
        else:
            np.save(pjoin(SAVE_DIR, MEAN_DIR,  f'eval_{object_name}_Mean.npy'), Mean)
            np.save(pjoin(SAVE_DIR, STD_DIR,  f'eval_{object_name}_Std.npy'), Std)
    else:
        print(f'No .npy file found for object_type {object_name}')


def test_spatial_augmentation():
    """test recover_quats_and_offs"""
    other_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Anaconda/__Attack.bvh"
    tpos_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Anaconda/__Idle.bvh"
    object_type = "Anaconda"
    err_dict = {}
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(tpos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    motion, parents, max_joints = get_motion(other_bvh, FOOT_CONTACT_VEL_THRESH, object_type, ground_height, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, err_dict)
    augmented_motion, augmented_parents, joints_perm, permutations = spatial_augmentation(motion.copy(), parents)
    print("augmentations: "+str(permutations))
    print("original_shape: "+str(motion.shape))
    print("aug_shape: "+str(augmented_motion.shape))
    rots, offs, parents = recover_quats_and_offs_np(motion, motion.shape[1], parents)
    rots_aug, offs_aug, parents_aug = recover_quats_and_offs_np(augmented_motion, augmented_motion.shape[1], augmented_parents)
    print("original parents: "+str(parents))
    print("augmented parents: "+str(parents_aug))
    anim = Animation(rots, offs, Quaternions.id(offs.shape[0]), offs[0], parents)
    anim_aug = Animation(rots_aug, offs_aug, Quaternions.id(offs_aug.shape[0]), offs_aug[0], parents_aug)
    positions = positions_global(anim)
    positions_aug = positions_global(anim_aug)
    plot_single_frame(positions, 0, None, anim.parents, TEST_OUTPUT_DIR, "test_spatial_augmentation_" + object_type + "_original.png")
    plot_single_frame(positions_aug, 0, None, anim_aug.parents,TEST_OUTPUT_DIR, "test_spatial_augmentation_" + object_type + "_augmented.png")


def test_positions_mismatch():
    bvh_path = "dataset/truebones/zoo/Truebone_Z-OO/Trex/__chase_roar_left.bvh"
    old_anim, _1, _2 = BVH.load(bvh_path)
    plot_3d_motion(old_anim, "dataset/truebones/before_replacement.mp4", figsize=(10, 10), fps=30, radius=10, title = "recovered_anim")
    old_anim.positions[:, 1:, :] = old_anim.offsets[1:][None, :].repeat(old_anim.positions.shape[0], axis=0) 
    old_anim, root_pose_init_xz_, ground_height_, scale_factor_ = process_anim(old_anim, 'Trex')
    plot_3d_motion(old_anim, "dataset/truebones/after_replacement.mp4", figsize=(10, 10), fps=30, radius=10, title = "recovered_anim")


def test_recover_motion():
    """test recover_quats_and_offs. To be able to run it, uncomment the new_anim return value on line 262 in get_motion func"""
    other_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Scorpion/__Defend.bvh"
    tpos_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Scorpion/__TPOSE.bvh"
    object_type = "Scorpion"
    err_dict = {}
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots , names, tpos_anim= get_common_features_from_T_pose(tpos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    motion, parents, orig_anim = get_motion(other_bvh, FOOT_CONTACT_VEL_THRESH, object_type, ground_height, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, err_dict)
    rots, offs, parents = recover_quats_and_offs(motion[0])
    anim = Animation(rots, offs, Quaternions.id(offs.shape[0]), offs[0], parents)
    positions_orig = positions_global(orig_anim)
    positions = positions_global(anim)
    fig = plt.figure(figsize=(7, 7))
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    ax.view_init(90,-90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for joint, parent in enumerate(anim.parents[1:], start=1):
        ax.plot3D(positions[10,[joint, parent], 0], positions[10, [joint, parent], 1],
                positions[10, [joint, parent], 2], color="green", linewidth=2, linestyle='-', marker='o')

        ax.plot3D(positions_orig[10,[joint, parent], 0], positions_orig[10, [joint, parent], 1],
                positions_orig[10, [joint, parent], 2], color="blue", linewidth=2, linestyle='-', marker='o')
    plt.savefig("dataset/truebones/"+object_type+"_both.png")


def test_recover_foot_contact():
    other_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Dragon/__Attack2.bvh"
    tpos_bvh = "dataset/truebones/zoo/Truebone_Z-OO/Dragon/__Tpose.bvh"
    object_type = "Dragon"
    err_dict = {}
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(tpos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    motion, parents, max_joint = get_motion(other_bvh, FOOT_CONTACT_VEL_THRESH, object_type, ground_height, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, err_dict)
    rots, offs, parents = recover_quats_and_offs(motion)
    foot_contact = recover_foot_contact(motion)
    anim = Animation(rots, offs, Quaternions.id(offs.shape[0]), offs[0], parents)
    positions = positions_global(anim)
    plot_single_frame(positions, 60, foot_contact, anim.parents, TEST_OUTPUT_DIR, object_type+"_foot_cont.png")


def augment_single_object(object_type):
    ## prepare
    os.makedirs(pjoin(AUGMENTED_SAVE_PREFIX , "augmented_motions"), exist_ok=True)
    os.makedirs(pjoin(AUGMENTED_SAVE_PREFIX,  "augmented_texts"), exist_ok=True)
    all_motions_of_object = [f for f in os.listdir(pjoin(SAVE_DIR, MOTION_DIR)) if f.lower().startswith(object_type.lower() + '_')]
    files_counter = 0
    frames_counter = 0 
    for motion_npy in all_motions_of_object:
        print("Augmenting file: " + motion_npy.split('.')[0])
        motion = np.load(pjoin(SAVE_DIR, MOTION_DIR, motion_npy))
        text_file = open(pjoin(SAVE_DIR, TEXT_DIR, motion_npy.split('.')[0] + '.txt'), 'r')
        text = text_file.readlines()[0] 
        text_file.close()
        prev_files_counter = files_counter
        files_counter, frames_counter = motion_augmentation(files_counter, frames_counter, motion, text, 40, AUGMENTATIONS_PER_MOTION)
    print('Created %d augmented versions.' %(files_counter - prev_files_counter))
    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))

def test_process_object(object_type):
    ## prepare
    os.makedirs(pjoin(TEST_OUTPUT_DIR, TEXT_DIR), exist_ok=True)
    os.makedirs(pjoin(TEST_OUTPUT_DIR, MOTION_DIR), exist_ok=True)
    
    
    os.makedirs(pjoin(TEST_OUTPUT_DIR, NORMALIZED_BONE_LENGTH_DIR), exist_ok=True)
    os.makedirs(pjoin(TEST_OUTPUT_DIR, ANIMATIONS_DIR), exist_ok=True)
    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, dict(), save_dir=TEST_OUTPUT_DIR)
    np.save(pjoin(TEST_OUTPUT_DIR, "cond.npy"), object_cond)

def test_recover_from_ric(sample):
    tpos_bvh = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/mixamo/MouseyNoFingers/Idle.bvh"
    other_bvh = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/mixamo/MouseyNoFingers/Catwalk Walk Forward Arc 90L.bvh"
    object_type = "MouseyNoFingers"
    err_dict = {}
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(tpos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    motion, parents, max_joints = get_motion(other_bvh, FOOT_CONTACT_VEL_THRESH, object_type, ground_height, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, err_dict)
    positions, parents = recover_from_ric_np(sample, parents)
    plot_single_frame(positions, 10, None, parents, TEST_OUTPUT_DIR, "test_recover_from_ric.png")
    
    

def hml_rep_to_truebones_tensor(data, parents=t2m_kinematic_parents, offsets=None):
    # motion.shape n_frames, 263
    # new representation
    # joint #0 = r_velocity||l_velocity||root_y||r_velocity, l_velocity, root_root_local_velocity
    # all other joints = ric || rotation || local_velocity || offset || foot_contact 
    if offsets is None:
        feature_len = 25
        feet_ind = 12
    else:
        feature_len = 31
        feet_ind = 15
    n_joints = 22
    if len(data.shape) == 1:
        data = data[None, :]
    if len(data.shape) == 2:
        n_frames = data.shape[0]
        truebones_rep = np.zeros((n_frames, n_joints , feature_len))
    else: # len(data.shape) == 3 (bs, frames, 263)
        bs, n_frames = data.shape[:2]
        truebones_rep = np.zeros((bs, n_frames, n_joints , feature_len))

    left_foot = [7, 10]
    right_foot = [8, 11]
    
    positions = data[...,4: 67].reshape(n_frames, n_joints - 1, 3)
    rotations = data[...,67: 193].reshape(n_frames, n_joints - 1, 6)
    velocity = data[...,193: 259].reshape(n_frames, n_joints, 3)
    feet_contact = np.zeros((n_frames, n_joints))
    feet_contact[..., left_foot[0]] = data[..., 259]
    feet_contact[..., left_foot[1]] = data[..., 260]
    feet_contact[..., right_foot[0]] = data[..., 261]
    feet_contact[..., right_foot[1]] = data[..., 262]
    root_data = np.concatenate((data[..., :4], velocity[..., 0, :]), axis=-1)
    truebones_rep[..., 0, :7] = root_data
    truebones_rep[..., 1:, :3] = positions
    truebones_rep[..., 1:, 3:9] = rotations
    truebones_rep[..., 1:, 9:12] = velocity[..., 1:, :]
    if offsets is not None:
        truebones_rep[..., 1:, 12:15] = offsets[1:][None, :]
    truebones_rep[... ,feet_ind] = feet_contact
    for j, p in enumerate(parents[1:], start=1):
        truebones_rep[..., j, feet_ind + 1:] = truebones_rep[..., p, :feet_ind]
    return truebones_rep

def truebones_tensor_to_hml_rep_np(data):
    # assumes data in hml original joints order 
    # data.shape (bs, frames, joints, feature_len)
    left_foot = [7, 10]
    right_foot = [8, 11]
    n_frames = data.shape[1]
    n_joints = data.shape[2]
    bs = data.shape[0]
    hml_feature_len = 263
    hml_rep = np.zeros((bs, n_frames , hml_feature_len))
    root_data = data[:, :, 0, :4] # bs, n_frames, 4
    ric_data = data[:, :, 1:, :3].reshape((bs, n_frames, 3 * (n_joints - 1))) # bs, n_frames, 3*(n_joints-1)
    rot_data = data[:, :, 1:, 3:9].reshape((bs, n_frames, 6 * (n_joints - 1))) # bs, n_frames, 6*(n_joints-1)
    local_vel = np.concatenate([data[:,:, 0, 4:7] , data[:, :, 1:, 9:12].reshape((bs, n_frames, 3 * (n_joints - 1)))], axis = -1) # bs, n_frames, n_joints*3
    feet_l = data[:, :, left_foot, 12].reshape((bs, n_frames, 2))
    feet_r = data[:, :, right_foot, 12].reshape((bs, n_frames, 2))
    hml_rep = root_data
    hml_rep = np.concatenate([hml_rep, ric_data], axis=-1)
    hml_rep = np.concatenate([hml_rep, rot_data], axis=-1)
    hml_rep = np.concatenate([hml_rep, local_vel], axis=-1)
    hml_rep = np.concatenate([hml_rep, feet_l, feet_r], axis=-1)
    return hml_rep

def truebones_tensor_to_hml_rep(data):
    # assumes data in hml original joints order 
    # data.shape (bs, joints, feature_len, frames)
    left_foot = [7, 10]
    right_foot = [8, 11]
    mat = data.permute(0, 3, 1, 2)
    n_frames = mat.shape[1]
    n_joints = mat.shape[2]
    bs = mat.shape[0]
    hml_feature_len = 263
    hml_rep = torch.zeros(bs, n_frames , hml_feature_len)
    root_data = mat[:, :, 0, :4] # bs, n_frames, 4
    ric_data = mat[:, :, 1:, :3].reshape((bs, n_frames, 3 * (n_joints - 1))) # bs, n_frames, 3*(n_joints-1)
    rot_data = mat[:, :, 1:, 3:9].reshape((bs, n_frames, 6 * (n_joints - 1))) # bs, n_frames, 6*(n_joints-1)
    local_vel = torch.cat([mat[:,:, 0, 4:7] , mat[:, :, 1:, 9:12].reshape((bs, n_frames, 3 * (n_joints - 1)))], dim = -1) # bs, n_frames, n_joints*3
    feet_l = mat[:, :, left_foot, 15].reshape((bs, n_frames, 2))
    feet_r = mat[:, :, right_foot, 15].reshape((bs, n_frames, 2))
    hml_rep = root_data
    hml_rep = torch.cat([hml_rep, ric_data], dim=-1)
    hml_rep = torch.cat([hml_rep, rot_data], dim=-1)
    hml_rep = torch.cat([hml_rep, local_vel], dim=-1)
    hml_rep = torch.cat([hml_rep, feet_l, feet_r], dim=-1)
    return hml_rep

def create_obj_adjucentcy_mask(parents):
    n = len(parents)
    mask = np.eye(n)
    for i in range(n):
        parent = parents[i]
        if parent ==  -1:
            siblings = []
        else:
            siblings = [j for j in range(n) if parents[j] == parent and j != i]
        children = [j for j in range(n) if parents[j] == i]
        mask[i, siblings] = 1.0
        if parent != -1:
            mask[i, parent] = 1.0
        mask[i, children] = 1.0
    return mask

def create_topology_edge_relations(parents, max_path_len = 10): # joint j+1 contains len(j, j+1)
    edge_types = {'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5, 'ts_token_conn': 6}
    topo_types = {'far': max_path_len + 1, 'ts_token_conn': max_path_len + 2}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation'] 
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            """Update edge type"""
            edge_type = edge_types['no_relation']
            if i == j: #self
                edge_type = edge_types['self'] 
            elif parent_j == i: #child
                ee=False
                edge_type = edge_types['child']
            elif j == parent: #parent
                edge_type = edge_types['parent'] 
            elif parent_j == parent: #sibling
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            """Update path length type"""
            
            if i == j:
                topo_rel[i, j] = 0      
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i: # parent-child relation
                topo_rel[i, j] = 1
            else: #any other 
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']
            
    topo_rel[topo_rel > max_path_len] = topo_types['far']
    return edge_rel, topo_rel

def create_obj_adjucentcy_mask_with_ee_and_root(parents):
    ee = [i for i in range(len(parents)) if i not in parents]
    mask = create_obj_adjucentcy_mask(parents)
    mask[:, 0] = 1.0
    mask[:, ee] = 1.0
    return mask

def create_obj_adjucentcy_mask_kinematic_chains(parents, full=False):
    n = len(parents)
    mask = np.eye(n)
    for i in range(1, n):
        mask[i, :i] = mask[parents[i], :i] 
    if full:
        for i in range(n-1, -1, -1):
            mask[parents[i]] = np.logical_or(mask[i], mask[parents[i]]).astype(int)
    return mask

def create_obj_adjucentcy_mask_new(parents):
    n = len(parents)
    mask = np.zeros((n,n))
    for i in range(n):
        parent = parents[i]
        # if parent ==  -1:
        #     siblings = []
        # else:
        #     siblings = [j for j in range(n) if parents[j] == parent and j != i]
        children = [j for j in range(n) if parents[j] == i]
        # mask[i, siblings] = 0.1
        # if parent != -1:
        #     mask[i, parent] = 1.0
        if len(children) == 0:
            children = [0]
        mask[i, children] = 1.0

    return mask

def test_hml_as_truebones_rep():
    example_id = "000025"
    save_dir2 = '/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/new_joint_vecs'
    example_data = np.load(os.path.join(save_dir2, example_id + '.npy'))
    hml_as_truebones_tensor = hml_rep_to_truebones_tensor(example_data, t2m_kinematic_parents, smpl_offsets)
    positions, parents = recover_from_ric_np(hml_as_truebones_tensor, t2m_kinematic_parents)
    plot_single_frame(positions, 10, None, parents , TEST_OUTPUT_DIR, "test_hml_as_truebones_rep.png")

def test_hml_full_augmentation_as_truebones_rep():
    example_id = "000025"
    save_dir2 = '/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/new_joint_vecs'
    example_data = np.load(os.path.join(save_dir2, example_id + '.npy'))
    hml_as_truebones_tensor = hml_rep_to_truebones_tensor(example_data, t2m_kinematic_parents, smpl_offsets)
    new_parents, permutation, num_of_perms = parents_augmentation_full(t2m_kinematic_parents)
    augmented_motion = hml_as_truebones_tensor[:, permutation]
    positions, parents = recover_from_ric_np(augmented_motion, new_parents)
    orig_positions, orig_parents = recover_from_ric_np(hml_as_truebones_tensor, t2m_kinematic_parents)
    plot_general_skeleton_3d_motion(pjoin(TEST_OUTPUT_DIR, "full_augmentation_hml.mp4"), new_parents, positions, dataset="truebones", title="test_hml_full_augmentation", fps=20)
    plot_general_skeleton_3d_motion(pjoin(TEST_OUTPUT_DIR, "orig_hml.mp4"), orig_parents, orig_positions, dataset="truebones", title="test_hml_full_augmentation", fps=20)

def add_hml_to_truebones_dataset(num_of_samples, motion_rep="Full"):
    hml_files_list = os.listdir(pjoin(HML_DATA_PATH, HML_MOTION_DIR))
    hml_files_count = len(hml_files_list)
    jump = hml_files_count // num_of_samples
    tensors_list = list()
    for i in range(0, hml_files_count, jump):
        sample = np.load(pjoin(HML_DATA_PATH, HML_MOTION_DIR, hml_files_list[i]))
        motion_data = hml_rep_to_truebones_tensor(sample, t2m_kinematic_parents, smpl_offsets)
        tensors_list.append(motion_data)
        augmented_motion, new_parents, joints_perm, permutations = spatial_augmentation_hard(motion_data, t2m_kinematic_parents)
        file_name = hml_files_list[i].split('.')[0]
        text_file = open(pjoin(HML_DATA_PATH, TEXT_DIR, file_name + '.txt'), 'r')
        lines = text_file.readlines()
        line_ind = random.randint(0, len(lines) - 1)
        text = lines[line_ind].rstrip() 
        text_file.close()
        text = augmentation_text_hml(text, new_parents, joint_names, joints_perm)
        new_file_name = "HARD_AUG_HML"+file_name
        if motion_rep=="Full":
            full_rep_aug = compact_rep_to_full_rep(augmented_motion, new_parents)
        np.save(pjoin(AUGMENTED_SAVE_PREFIX+"_motions", new_file_name + '.npy'), full_rep_aug)
        new_text_file = open(pjoin(AUGMENTED_SAVE_PREFIX+"_texts", new_file_name+'.txt'), "w")
        n = new_text_file.write(text)
        text_file.close()
    save_mean_std(tensors_list, "Human")

def augment_hml_motion(hml_motion):
    motion_data = hml_rep_to_truebones_tensor(hml_motion, t2m_kinematic_parents)
    augmented_motion, new_parents, joints_perm, permutations = spatial_augmentation_hard(motion_data, t2m_kinematic_parents)
    return augmented_motion, new_parents, joints_perm

def mix_joints(motion, parents):
    augmented_motion, new_parents, joints_perm, permutations = spatial_augmentation_hard(motion, parents)
    return augmented_motion, new_parents, joints_perm


def create_split_files():
    all_motions_ids = [f.split('.')[0] for f in os.listdir(AUGMENTED_SAVE_PREFIX+"_motions")]
    scorpion_samples = np.arange(5446, 5492)
    flamingo_samples = np.arange(0, 24)
    
    test_samples = [s for i, s in enumerate(all_motions_ids) if i in scorpion_samples or i in flamingo_samples]
    train_samples = [s for s in all_motions_ids if s not in test_samples]

    train_file = open(pjoin(SAVE_DIR, "train.txt"), "w")
    n = train_file.writelines(line + '\n' for line in train_samples)
    train_file.close()

    test_file = open(pjoin(SAVE_DIR, "test.txt"), "w")
    n = test_file.writelines(line + '\n' for line in test_samples)
    test_file.close()

def create_hml_split_file():
    hml_motions_ids = [f.split('.')[0] for f in os.listdir(AUGMENTED_SAVE_PREFIX+"_motions") if f.startswith('HARD_AUG')]
    test_file = open(pjoin(SAVE_DIR, "hml_test.txt"), "w")
    n = test_file.writelines(line + '\n' for line in hml_motions_ids)
    test_file.close()

def create_truebones_split_file():
    truebones_motions_ids = [f.split('.')[0] for f in os.listdir(pjoin(SAVE_DIR, MOTION_DIR)) if f.endswith('.npy')]
    test_file = open(pjoin(SAVE_DIR, "test.txt"), "w")
    n = test_file.writelines(line + '\n' for line in hml_motions_ids)
    test_file.close()

def per_obj_adjucency_mask():
    save_dir = pjoin(SAVE_DIR, "adjucency_masks")
    os.makedirs(save_dir, exist_ok=True)
    objects = [ d for d in os.listdir(DATA_DIR) if os.path.isdir(pjoin(DATA_DIR, d))]
    for object_type in objects:
        bvh_files= [pjoin(DATA_DIR, object_type, f) for f in os.listdir(pjoin(DATA_DIR, object_type)) if f.lower().endswith('.bvh')]
        if len(bvh_files) == 0:
            continue
        ## get t-pos bvh
        t_pos_path = None
        for f in bvh_files:
            if "tpos" in f.lower():
                t_pos_path = f
                break
        if t_pos_path is not None:
            bvh_files.remove(t_pos_path)
        else: #choose some other motion to be treated as tpos 
            for f in bvh_files:
                if "idle" in f.lower():
                    t_pos_path = f
                    break
        if t_pos_path is None:
            t_pos_path = bvh_files[0]
        t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pos_path)
        parents = t_pose_anim.parents
        mask = create_obj_adjucentcy_mask(parents)
        np.save(pjoin(save_dir, object_type + '.npy'), mask)

def add_human_adj_mask():
    save_dir = pjoin(SAVE_DIR, "adjucency_masks")
    mask = create_obj_adjucentcy_mask_kinematic_chains(t2m_kinematic_parents, full=False)
    np.save(pjoin(save_dir, "Human_kinematic_mask_partial" + '.npy'), mask)

def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

def parents2kinchains(parents, policy = 'h_first'):
    chains = list()
    children_dict = {i:[] for i in range(len(parents))}
    for j,p in enumerate(parents[1: ], start=1):
        if policy == 'h_first':
            reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)
    recursion_kinchains([], 0, children_dict, chains, policy)
    return chains


def recursion_kinchains(chain, j, children_dict, chains, policy):
    children = children_dict[j]
    if len(children) == 0: #ee
        chain.append(j)
        chains.append(chain) 
    elif len(children) == 1:
        chain.append(j)
        recursion_kinchains(chain, children[0], children_dict, chains, policy)
    else:
        chain.append(j)
        if policy == 'h_first':
            main_child = max(children)
        else:
            main_child = min(children)
        for child in children:
            if child == main_child:
                recursion_kinchains(chain, child, children_dict, chains, policy)
            else:
                recursion_kinchains([j], child, children_dict, chains, policy)
            
def create_vis_for_kinchain_labeling():
    objects = [ d for d in os.listdir(DATA_DIR) if os.path.isdir(pjoin(DATA_DIR, d))]
    for object_type in objects:
        bvh_files= [pjoin(DATA_DIR, object_type, f) for f in os.listdir(pjoin(DATA_DIR, object_type)) if f.lower().endswith('.bvh')]
        if len(bvh_files) == 0 or FACE_JOINTS[object_type] == []:
            continue
        ## get t-pos bvh
        t_pos_path = None
        for f in bvh_files:
            if "tpos" in f.lower():
                t_pos_path = f
                break
        if t_pos_path is not None:
            bvh_files.remove(t_pos_path)
        else: #choose some other motion to be treated as tpos 
            for f in bvh_files:
                if "idle" in f.lower():
                    t_pos_path = f
                    break
        if t_pos_path is None:
            t_pos_path = bvh_files[0]
        root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots , names, tpos_anim= get_common_features_from_T_pose(t_pos_path, object_type, FOOT_CONTACT_HEIGHT_THRESH)
        anim, names = get_hml_aligned_anim(t_pos_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, {})
        positions = positions_global(anim)
        kinchains = parents2kinchains(anim.parents, object_policy(object_type))
        plot_single_frame_kinchains(positions, 0, kinchains, TEST_OUTPUT_DIR, object_type+"_kinchains.png", names, FACE_JOINTS[object_type])
        kinchains

def test_mean_std_norm():
    hml_mean = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/Mean.npy")
    hml_std = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/Std.npy")
    hml_motion = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/new_joint_vecs/000010.npy")
    normalized1 = (hml_motion - hml_mean) / hml_std
    # normalized1 = hml_rep_to_truebones_tensor(normlized_hml_before_transform)
    
    tb_mean = hml_rep_to_truebones_tensor(hml_mean[None,:])
    tb_std = hml_rep_to_truebones_tensor(hml_std[None, :])
    tb_motion = hml_rep_to_truebones_tensor(hml_motion)
    tb_motion = (tb_motion - tb_mean) / tb_std
    tb_motion = np.nan_to_num(tb_motion)
    normalized2 = truebones_tensor_to_hml_rep_np(tb_motion[None, :])
    normalized2

def get_anim(t_pos_bvh, motion_bvh, object_type):
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(t_pos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    new_anim, names = get_hml_aligned_anim(motion_bvh, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, {})
    return new_anim, names

def test_convert_to_kinematic_rots():
    # plot_general_skeleton_3d_motion(pjoin(TEST_OUTPUT_DIR, "test_convert_to_kinematic_rots.mp4"), ostrich_parents, orig_pos, dataset="truebones", title="test_convert_to_kinematic_rots", fps=20)
    t_pos_bvh = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/mixamo/MouseyNoFingers/Idle.bvh"
    motion_bvh = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/mixamo/MouseyNoFingers/Jumping.bvh"
    object_type = "MouseyNoFingers"
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim = get_common_features_from_T_pose(t_pos_bvh, object_type, FOOT_CONTACT_HEIGHT_THRESH)
    motion, parents, max_joints = get_motion(motion_bvh, FOOT_CONTACT_VEL_THRESH, object_type, offsets.shape[0], root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, {})
    new_anim, names = get_anim(t_pos_bvh, motion_bvh, object_type)
    normalized_offsets = normalize(new_anim.offsets.copy(), axis=1)
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    skel = Skeleton(torch.from_numpy(normalized_offsets), kinematic_chains, "cpu")
    skel.set_offset(torch.from_numpy(new_anim.offsets))
    positions = recover_from_rot_np(motion, skel)
    plot_general_skeleton_3d_motion(pjoin(TEST_OUTPUT_DIR, "test_convert_to_kinematic_rots_recovered_from_rot_mousey.mp4"), parents, positions.detach().cpu().numpy(), dataset="truebones", title="test_convert_to_kinematic_rots", fps=20)


def bvh_to_hml_vec(bvh_path, face_joints, foot_indices, foot_contact_vel_thresh):
    # face_joints -> [right hip, left hip, right shoulder, left shoulder]
    anim, names, frame_time = BVH.load(bvh_path)
    parents = anim.parents
    global_positions = positions_global(anim)
    frames, joints, _ = global_positions.shape
    offsets = anim.offsets.copy()
    normalized_offsets = normalize(offsets, axis=1)
    kinematic_chains = parents2kinchains(parents, object_policy("")) # use your function or manually define the kinematic chains
    skel = Skeleton(torch.from_numpy(normalized_offsets), kinematic_chains, "cpu")
    quat_params = Quaternions(skel.inverse_kinematics_np(global_positions, face_joints, smooth_forward=True))
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(quat_params)[:-1, 1:].reshape((frames - 1, (joints - 1) * 6))
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (global_positions[1:, 0] - global_positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    foot_contact = get_foot_contact(global_positions, foot_indices, foot_contact_vel_thresh)[:, foot_indices] 
    '''Get Joint Rotation Invariant Position Represention'''
    # local velocity wrt root coords system as described in get_rifke definition 
    ric_positions = get_rifke(global_positions, r_rot)[:-1, 1:].reshape((frames - 1, (joints - 1) * 3))
    root_y = global_positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    local_vel = (np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])).reshape((frames - 1, joints * 3))
    # root data shape 
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    feats = np.concatenate([root_data, ric_positions, cont_6d_params, local_vel, foot_contact], axis = -1)
    return feats
    
def hml_vec2mat(hml_vec):
    feet = [7, 10, 8, 11]
    n_frames = hml_vec.shape[0]
    n_joints = 22
    n_feats = 13
    hml_mat = np.zeros((n_frames, n_joints, n_feats))
    root_data = hml_vec[:, :4]
    linear_vel = hml_vec[:, 193:259].reshape((n_frames, n_joints, 3))
    ric_positions = hml_vec[:, 4: 67].reshape((n_frames, n_joints-1, 3))
    rotations = hml_vec[:, 67: 193].reshape((n_frames, n_joints-1, 6))
    fc = np.zeros((n_frames, n_joints))
    fc[:, feet] = hml_vec[:, 259:]
    hml_mat[:, 0, :4] = root_data
    hml_mat[:, 0, 4:7] = linear_vel[:, 0]
    hml_mat[:, 1:, :3] = ric_positions
    hml_mat[:, 1:, 3:9] = rotations
    hml_mat[:, 1:, 9:12] = linear_vel[:, 1:]
    hml_mat[:, :, 12] = fc
    return hml_mat    

def create_cond_dict_for_hml():
    object_cond = dict()
    positions = torch.zeros((1, 22, 3))
    positions[:, 0] = smpl_offsets[0][None, :]
    for j, p  in enumerate(t2m_kinematic_parents[1:], start=1):
        positions[:, j] = positions[:, p] + smpl_offsets[j][None, :]
    hml_vec, ground_positions, positions, l_velocity = hml_process(positions.numpy().repeat(2, axis=0), 0.002)
    hml_mat = hml_vec2mat(hml_vec)
    
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(t2m_kinematic_parents, max_path_len = MAX_PATH_LEN)
    object_cond['tpos_first_frame'] = hml_mat[0]
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    edge_len = list()
    for i in range(1, 22):
        edge_len.append(torch.norm(smpl_offsets[i] - smpl_offsets[t2m_kinematic_parents[i]], p=2, dim=0))
    normalized_edge_len = np.array(edge_len - HML_AVG_BONELEN) / np.std(edge_len)
    object_cond['normalized_edge_len'] = normalized_edge_len
    object_cond['object_type'] = "Human"
    object_cond['parents'] = np.array(t2m_kinematic_parents)
    object_cond['offsets'] = smpl_offsets.numpy()
    object_cond['joints_names'] = joint_names
    object_cond['kinematic_chains'] = t2m_kinematic_chain
    object_cond['mean'] = hml_vec2mat(np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/Mean.npy")[None, :])[0]
    object_cond['std'] = hml_vec2mat(np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/HumanML3D/Std.npy")[None, :])[0]
    object_cond['std'][object_cond['std'] == 0.0] = 1.0
    np.save(pjoin(SAVE_DIR, "hml_cond.npy"), {"Human": object_cond})
    

def remove_prefix(s):
        # Check if the string starts with any prefix and remove it
        if s.startswith('Sabrecat'):
            s = s[:-6]
        for prefix in REMOVE_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):]  
        return s 
    
def split_and_replace(s):
    splitted = re.split('(?=[A-Z]|_)', s)
    new_splitted = list()
    for part in splitted:
        # remove numbers and _
        clean_part = re.sub(r'[\d_]+', '', part)
        if clean_part == '':
            continue
        elif clean_part in ['L', 'l']:
            new_splitted.append("Left")
        elif clean_part in ['R', 'r']:
            new_splitted.append("Right")
        elif len(clean_part) == 1:
            continue
        elif clean_part in JAPANESE_WORDS.keys():
            clean_part = JAPANESE_WORDS[clean_part]
            new_splitted.append(clean_part)
        elif clean_part == 'Tai':
            clean_part = 'Tail'
            new_splitted.append(clean_part)
        else:
            new_splitted.append(clean_part)
    return ' '.join(new_splitted)     
            
            
def remove_joints_augmentation(data, removal_rate, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, normalized_edge_len, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['normalized_edge_len'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    removal_options = [j for j in ee if j not in possible_feet]
    remove_joints = sorted(random.sample(removal_options, math.ceil(len(removal_options) * removal_rate)), reverse=True)
    motion = np.delete(motion, remove_joints, axis=1)
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5    
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)
        
    for rj in remove_joints:
        parents[parents > rj] -= 1
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    normalized_edge_len = np.delete(normalized_edge_len, [j-1 for j in remove_joints], axis=0)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, normalized_edge_len, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std

def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, normalized_edge_len, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['normalized_edge_len'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    # added joint mut follow:
    # j has exactly 1 child 
    # j parent is not the root joint
    # j is not the root joint
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, normalized_edge_len, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
        
    add_j = random.choice(possible_joints_to_add)
    j_new_ind = parents.tolist().index(add_j)
    
    # motion features
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy() # rotations
    new_feats[..., 12] = j_feats[..., 12].copy() # feet 
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)
    
    # tpos features
    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy() # rotations
    tpos_new_feats[12] = tpos_j_feats[12] # feet 
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # mean features
    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy() # rotations
    mean_new_feats[12] = mean_j_feats[12] # feet 
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # std features
    std_new_feats = std[add_j].copy()
    
    # joints names embs features 
    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2
    
    # apply augmentation
    #motion
    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    #tpos_first_frame
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    #mean TODO: AUGMENT LIKE MOTION AND TPOS 
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    #std TODO: AUGMENT LIKE MOTION AND TPOS 
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    #joints_names_embs
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    # parents 
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    # topology conditions 
    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)
    
    # all others 
    normalized_edge_len = np.hstack([normalized_edge_len[:add_j], normalized_edge_len[add_j]/2, normalized_edge_len[add_j]/2, normalized_edge_len[add_j+1:]])
    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'
    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, normalized_edge_len, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains, mean_augmented, std_augmented

def _remove_prefix(s):
    # Check if the string starts with any prefix and remove it
    if s.startswith('Sabrecat'):
        s = s[:-6]
    for prefix in REMOVE_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]  
    return s 

def _split_and_replace(s):
    splitted = re.split('(?=[A-Z]|_)', s)
    new_splitted = list()
    for part in splitted:
        # remove numbers and _
        clean_part = re.sub(r'[\d_]+', '', part)
        if clean_part == '':
            continue
        elif clean_part in ['L', 'l']:
            new_splitted.append("Left")
        elif clean_part in ['R', 'r']:
            new_splitted.append("Right")
        elif len(clean_part) == 1:
            continue
        elif clean_part in JAPANESE_WORDS.keys():
            clean_part = JAPANESE_WORDS[clean_part]
            new_splitted.append(clean_part)
        elif clean_part == 'Tai':
            clean_part = 'Tail'
            new_splitted.append(clean_part)
        else:
            new_splitted.append(clean_part)
    return ' '.join(new_splitted)   


def quat_to_mat(qs):
    r = qs[..., 0]
    i = qs[..., 1]
    j = qs[..., 2]
    k = qs[..., 3]
    two_s = 2.0 / (qs * qs).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    rotations = o.reshape(qs.shape[:-1] + (3, 3))
    return rotations

    """ recover quaternions and positions from features for numpy only"""
def get_root_rot_pos_test(data):
    # data.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    
    root_rot_mat = quat_to_mat(r_rot_quat)
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3].float()
    '''Add Y-axis rotation to root position'''
    r_pos = torch.matmul(root_rot_mat.transpose(-1, -2), r_pos.unsqueeze(-1)).squeeze(-1)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    
    return root_rot_mat, r_pos
    
    """ recover xyz positions from ric (root relative positions) torch """
def recover_from_bvh_ric_test(data):
    r_rot, r_pos = get_root_rot_pos_test(data.permute(0, 3, 1, 2)[..., 0:1, 3:9])
    relative_positions = data.permute(0, 3, 1, 2)[..., 1:, :3]
    '''Add Y-axis rotation to local joints'''
    positions = torch.matmul(r_rot.transpose(-1, -2).repeat([1, 1, relative_positions.shape[2], 1, 1]) , relative_positions.unsqueeze(-1)).squeeze(-1)
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0]
    positions[..., 2] += r_pos[..., 2]
    '''Concate root and joints'''
    positions = torch.cat([r_pos, positions], dim=-2)
    return positions, r_rot, r_pos  

def recover_from_ric_test(data):
    #r_rot, r_pos = get_root_rot_pos_test(data.permute(0, 3, 1, 2)[..., 0:1, 3:9])
    relative_positions = data.permute(0, 3, 1, 2)[..., 1:, :3]
    '''Add Y-axis rotation to local joints'''
    '''Concate root and joints'''
    r_pos = torch.zeros(relative_positions.shape[0], relative_positions.shape[1], 1, relative_positions.shape[3])
    r_pos[..., 1] = data.permute(0, 3, 1, 2)[..., 0:1, 3]
    positions = torch.cat([r_pos, relative_positions], dim=-2)
    return positions  
if __name__ == "__main__":
    object_type = "Comodoa"
    out_path = "save/0_to_render2/more_unseen_comodoa/259"
    npy_path = "save/0_to_render2/more_unseen_comodoa/259/Comodoa__rep__#0.npy"
    motion = np.load(npy_path)
    # motion = motion[40:]
    cond = np.load("dataset/truebones/zoo/processed_bvh_rots/cond.npy", allow_pickle=True).item()[object_type]

    # positions from rots 
    offsets = cond['offsets']
    parents = cond['parents']
    joints_names = cond["joints_names"]
    global_positions_rots, out_anim0 = recover_from_bvh_rot_np(motion, parents, offsets)
    
    # positions from pos 
    global_positions_ric, _ = recover_from_bvh_ric_np(motion, parents)
    
    ## original offsets
    out_anim1, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets, iterations=150)
    
    ## first frame offsets
    out_anim2, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, iterations=150)
    
    ## frame 40 offsets
    offsets_40 = offsets_from_positions(global_positions_ric[40], parents)
    out_anim3, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets_40, iterations=150)
     
    ## frame 60 offsets
    offsets_60 = offsets_from_positions(global_positions_ric[60], parents)
    out_anim4, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets_60, iterations=150)
    
    ## frame 80 offsets
    offsets_80 = offsets_from_positions(global_positions_ric[80], parents)
    out_anim5, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets_80, iterations=150)
    
    ## frame 100 offsets
    offsets_100 = offsets_from_positions(global_positions_ric[100], parents)
    out_anim6, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets_100, iterations=150)
    
    ## frame 110 offsets
    offsets_110 = offsets_from_positions(global_positions_ric[110], parents)
    out_anim7, _1, _2 = animation_from_positions(positions=global_positions_ric, parents=parents, offsets=offsets_110, iterations=150)
    pref = os.path.basename(npy_path)[:-4]
    BVH.save(pjoin(out_path, f"{pref}_anim_from_rtos.bvh"), out_anim0, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_standard_offsets.bvh"), out_anim1, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame0_offsets.bvh"), out_anim2, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame40_offsets.bvh"), out_anim3, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame60_offsets.bvh"), out_anim4, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame80_offsets.bvh"), out_anim5, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame100_offsets.bvh"), out_anim6, names=joints_names)
    BVH.save(pjoin(out_path, f"{pref}_anim_from_ric_frame110_offsets.bvh"), out_anim7, names=joints_names)
    
    # ## read all conditions to a signle dict, save as cond.pkl and load at dataset initialization 
    # ## collect per each object_type its' mean, std, graph_dist, joints_relation, bones_len_diff
    # #create_data_samples()
    # #test_process_object("Ostrich")
    # REMOVE_PREFIXES = ["BN_Bip01","Bip01", "BN", "NPC", "jt", "Sabrecat", "Elk"]
    # JAPANESE_WORDS = {"momo":"Thigh", "sippo":"Tail", "mune":"Chest", "hiza":"Knee", "hara":"Stomach", "ashi":"Leg", "hiji": "Elbow", "koshi":"Hips", "te":"Hand", "kubi":"Neck", "atama":"Head", "ago":"Jaw", "kata":"Shoulder"}
    
    # object_types = FACE_JOINTS.keys()
    # motions_list = os.listdir(pjoin(SAVE_DIR, MOTION_DIR))
    # metadata_dict = dict()
    # for obj in object_types:
    #     obj_npys = [pjoin(SAVE_DIR, MOTION_DIR, f) for f in motions_list if f.startswith(f'{obj}_')]
    #     if len(obj_npys) == 0:
    #         continue
    #     total_frames = 0
    #     total_motions = len(obj_npys)
        
    #     for m in obj_npys:
    #         motion = np.load(m)
    #         n_frames = motion.shape[0]
    #         total_frames += n_frames
    #     print(f"{obj} has {total_motions} motions with {total_frames} in total")
    #     metadata_dict[obj] = {"total_frames": total_frames, "motions_count": total_motions} 
    # np.save(pjoin(SAVE_DIR, "per_object_n_frames.npy"),metadata_dict, allow_pickle=True)
            
            
             
    
    
    
    # motion = torch.from_numpy(np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_bvh_rots/motions/Cat_CAT_StretchYawnIdle_195.npy")).float()
    # cond = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_bvh_rots/cond.npy", allow_pickle=True).item()
    # gen_dir = "/home/dcor/inbargat1/multi-skeleton-mdm/save/all_random_pe2_trans.dec_dataset_truebones_bs_8_latentdim_128/samples_all_random_pe2_trans.dec_dataset_truebones_bs_8_latentdim_128_000599998_seed10"
    # out_dir = "/home/dcor/inbargat1/multi-skeleton-mdm/save/eval/bvh"
    # os.makedirs(out_dir, exist_ok=True)
    # all_npy = [pjoin(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(".npy")]
    # for i, f_path in enumerate(all_npy):
    #     positions = np.load(f_path)
    #     object_type = os.path.basename(f_path)[18:].split("_")[0]
    #     print(object_type)
    #     parents = cond[object_type]["parents"]
    #     offsets = cond[object_type]["offsets"]
    #     joints_names = cond[object_type]["joints_names"]
    #     anim, _1, _2 = animation_from_positions(positions=positions, parents=parents, offsets=offsets, iterations=150)
    #     BVH.save(pjoin(out_dir, f"{object_type}__{i}.bvh"), anim, joints_names)
    # positions, parents = recover_from_bvh_ric_np(motion, cond["parents"])
    #plot_general_skeleton_3d_motion("text_recover.mp4", cond['parents'], positions.detach().cpu().numpy()[0], dataset="truebones", title="", fps=20, face_joints=FACE_JOINTS["Cat"], fc=None)
    
    #create_data_samples()
    # motion = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed/motions/Stego___Attack3_897.npy")
    # object_type="Stego"
    # data = {
    #         "motion": motion,
    #         "length": motion.shape[0],
    #         "object_type": "Stego",
    #         "parents": cond["parents"],
    #         "joints_graph_dist": cond["joints_graph_dist"],
    #         "joints_relations": cond["joint_relations"],
    #         "normalized_edge_len": cond["normalized_edge_len"],
    #         "tpos_first_frame": cond["tpos_first_frame"],
    #         "offsets": cond["offsets"],
    #         "joints_names_embs": None,
    #         "kinematic_chains": cond["kinematic_chains"]
    #     }
        
    # add_joint_augmentation(data, cond['mean'], cond['std'])
    # data = get_dataset_loader(name="truebones", batch_size=2, num_frames=40, debug=True, temporal_window=31, hml=False, t5_name='t5-base')
    
    # for motion, cond in data:
    #     # recover both motions for sanity check 
    #     motions = motion.detach().cpu().numpy().transpose(0, 3, 1, 2)
    #     parents = cond['y']['parents']
    #     object_type = cond['y']['object_type']
    #     object_type1, add_j1 = object_type[0].split('__')
    #     object_type2, add_j2 = object_type[1].split('__')
    #     j1, j2 = cond['y']['n_joints'].tolist()
    #     cond_dict = data.dataset.t2m_dataset.cond_dict
        
    #     add_j1 = int(add_j1)
    #     mean1 = cond_dict[object_type1]['mean']
    #     std1 = cond_dict[object_type1]['std']
    #     mean1_j_feats = mean1[add_j1]
    #     mean1_p_feats = mean1[add_j1-1]
    #     mean1_new_feats = ((mean1_j_feats + mean1_p_feats)/2)
    #     mean1 = np.vstack([mean1[:add_j1], mean1_new_feats[None], mean1[add_j1:]])
    #     std1_j_feats = std1[add_j1]
    #     std1_p_feats = std1[add_j1-1]
    #     std1_new_feats = ((std1_j_feats + std1_p_feats)/2)
    #     std1 = np.vstack([std1[:add_j1], std1_new_feats[None], std1[add_j1:]])
        
    #     add_j2 = int(add_j2)
    #     mean2 = cond_dict[object_type2]['mean']
    #     std2 = cond_dict[object_type2]['std']
    #     mean2_j_feats = mean2[add_j2]
    #     mean2_p_feats = mean2[add_j2-1]
    #     mean2_new_feats = ((mean2_j_feats + mean2_p_feats)/2)
    #     mean2 = np.vstack([mean2[:add_j2], mean2_new_feats[None], mean2[add_j2:]])
    #     std2_j_feats = std2[add_j2]
    #     std2_p_feats = std2[add_j2-1]
    #     std2_new_feats = ((std2_j_feats + std2_p_feats)/2)
    #     std2 = np.vstack([std2[:add_j2], std2_new_feats[None], std2[add_j2:]])
        
    #     m1 = (motions[0, :, :j1] * std1[None]) + mean1[None]
    #     m2 = (motions[1, :, :j2] * std2[None]) + mean2[None]
    #     positions1, parents1 = recover_from_ric_np(m1, parents[0][:j1].tolist())
    #     positions2, parents2 = recover_from_ric_np(m2, parents[1][:j2].tolist())
    #     fc1 = [[j for j in range(len(parents1)) if motions[0,f, j , 12] != 0] for f in range(motions[0].shape[0])]
    #     fc2 = [[j for j in range(len(parents2)) if motions[1,f, j , 12] != 0] for f in range(motions[1].shape[0])]
    #     plot_general_skeleton_3d_motion("motion1.mp4", parents1, positions1, dataset="truebones", title="", fps=20, face_joints=[0,1,2,3], fc=fc1)
    #     plot_general_skeleton_3d_motion("motion2.mp4", parents2, positions2, dataset="truebones", title="", fps=20, face_joints=[0,1,2,3], fc=fc2)
        
        
            
    
    
    
    
    
    
        
        