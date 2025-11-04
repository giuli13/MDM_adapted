import numpy as np
import torch

def neighbor_joint(parents, threshold):
    n_joint = len(parents)
    dist_mat = np.empty((n_joint, n_joint), dtype=int)
    dist_mat[:, :] = 100000
    for i, p in enumerate(parents):
        dist_mat[i, i] = 0
        if i != 0:
            dist_mat[i, p] = dist_mat[p, i] = 1

    """
    Floyd's algorithm
    """
    for k in range(n_joint):
        for i in range(n_joint):
            for j in range(n_joint):
                dist_mat[i, j] = min(dist_mat[i, j], dist_mat[i, k] + dist_mat[k, j])

    neighbor_list = []
    for i in range(n_joint):
        neighbor = []
        for j in range(n_joint):
            if dist_mat[i, j] <= threshold:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    return neighbor_list

def first_half_pe(parents, depth_list):
    return [j if depth_list[j] % 2 == 0 else p for (j, p) in enumerate(parents)]

def second_half_pe(parents, depth_list):
    return [p if depth_list[j] % 2 == 0 else j for (j, p) in enumerate(parents)]

def depth_list(parents):
    sorted_deps = sorted(range(len(parents)), key=lambda k: parents[k])
    depth_list = [0 for i in range(len(parents))]
    for j in sorted_deps:
        if (parents[j] == -1):
            depth_list[j] = 0
        else:
            depth_list[j] = depth_list[parents[j]] + 1
    return depth_list


def ancestors_indices(parents, joint_index):
    if joint_index < 0 or joint_index > len(parents) - 1:
        return -1
    ancestors = []
    cur_ind = joint_index
    ancestors.append(cur_ind)
    while cur_ind > 0:
        cur_ind = parents[cur_ind]
        ancestors.append(cur_ind)
    return ancestors


def skeleton_mask(parents):
    joints_n = len(parents)
    mask = np.zeros((1,joints_n, joints_n)).astype('uint8')
    for j in range(joints_n):
        ancestors = ancestors_indices(parents, j)
        mask[0, j, ancestors] = 1
    return torch.from_numpy(mask)

def actual_joints_skeleton_mask(max_joints, bs_n_joints):
    bs_size = bs_n_joints.shape[0]
    mask = torch.zeros(bs_size,1,1,max_joints)
    # mask = np.zeros((bs_n_joints,1,1,max_joints)).astype('uint8')
    for b in range(bs_size):
        mask[b, 0, 0, :bs_n_joints[b]] = 1.
    return mask

def get_inv_perm(joints_perm):
    inv_perm = [0 for i in range(len(joints_perm))]
    for i, j in enumerate(joints_perm):
        inv_perm[j] = i
    return inv_perm