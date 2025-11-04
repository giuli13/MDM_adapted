import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from functools import partial
import os

if (ffmpeg_path := os.environ.get('FFMPEG_PATH','')) != '':
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    

    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax
    
    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        # used_colors = colors_blue if index in gt_frames else colors
        # for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
        #     if i < 5:
        #         linewidth = 4.0
        #     else:
        #         linewidth = 2.0
            
        #     ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
        #               color=color)
        # #         print(trajec[:index, 0].shape)
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            for j in range(1, len(chain)):
                bone_dist = math.sqrt(((data[index, chain[j]]-data[index, chain[j-1]])**2).sum())
                linewidth = 20 * bone_dist
                ax.plot3D(data[index, [chain[j-1], chain[j]], 0], data[index, [chain[j-1], chain[j]], 1], data[index, [chain[j-1], chain[j]], 2], linewidth=linewidth, color=color, solid_capstyle='round')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()

def plot_general_skeleton_3d_motion(save_path, parents, joints, title, dataset, figsize=(7, 7), fps=120, radius=5, face_joints = [], fc = None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set(xlim3d=(-radius / 2, radius / 2), xlabel='X')
        ax.set(ylim3d=(0, radius), ylabel='Y')
        ax.set(zlim3d=(-radius / 3., radius * 2 / 3.), zlabel='Z')
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax: Axes3D = fig.add_subplot(projection="3d") # type: ignore
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index, color):
        ax.clear()
        ax.view_init(elev=90, azim=-90)
        ax.set_box_aspect(None, zoom=1.5)
        color = color
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color=color, solid_capstyle='round')
            if joint in face_joints:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='blue', marker='o')
            if fc is not None and joint in fc[index]:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='green', marker='o')
        
        # plt.axis('off')
                ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z')
        ax.set_xticks(ax.get_xticks(), [])
        ax.set_yticks(ax.get_yticks(), [])
        ax.set_zticks(ax.get_zticks(), [])

    ani = FuncAnimation(fig, partial(update, color="red"), frames=frame_number, interval=1000 / fps)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
    
    

def plot_alignment(save_path, parents_1, joints_1, parents_2, joints_2, title, dataset_1, dataset_2, figsize=(7, 7), fps=120, radius=5, alignment=None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data_1 = joints_1.copy().reshape(len(joints_1), -1, 3)
    data_2 = joints_2.copy().reshape(len(joints_2), -1, 3)
    # preparation related to specific datasets
    if dataset_1 == 'kit':
        data_1 *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset_1 in ['humanml', 'truebones', 'humanml_mat']:
        data_1 *= 1.3  # scale for visualization
    elif dataset_1 in ['humanact12', 'uestc']:
        data_1 *= -1.5 # reverse axes, scale for visualization
    
        
    if dataset_2 == 'kit':
        data_2 *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset_2 in ['humanml', 'truebones', 'humanml_mat']:
        data_2 *= 1.3  # scale for visualization
    elif dataset_2 in ['humanact12', 'uestc']:
        data_2 *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    min_1 = data_1.min(axis=0).min(axis=0)
    min_2 = data_2.min(axis=0).min(axis=0)
    max_1 = data_1.max(axis=0).max(axis=0)
    max_2 = data_2.max(axis=0).max(axis=0)
    MINS = np.array([min(min_1[i], min_2[i]) for i in range(3)])
    MAXS = np.array([max(max_1[i], max_2[i]) for i in range(3)])
    frame_number_1 = data_1.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data_1[:, :, 1] -= height_offset
    trajec = data_1[:, 0, [0, 2]]

    # data_1[..., 0] -= data_1[:, 0:1, 0]
    data_1[..., 2] -= data_1[:, 0:1, 2]
    
    frame_number_2 = data_2.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data_2[:, :, 1] -= height_offset
    trajec = data_2[:, 0, [0, 2]]

    data_2[..., 0] -= data_2[:, 0:1, 0]
    data_2[..., 2] -= data_2[:, 0:1, 2]


    def update_1(index):
        ax.clear()
        ax.view_init(elev=90, azim=-90)
        ax.dist = 7.5
        color = "green"
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents_1[1:], start=1):
            ax.plot3D(data_1[index, [joint, parent], 0], data_1[index, [joint, parent], 1], data_1[index, [joint, parent], 2], color="green", solid_capstyle='round')
        for joint, parent in enumerate(parents_2[1:], start=1):
            ax.plot3D(data_2[index, [joint, parent], 0], data_2[index, [joint, parent], 1], data_2[index, [joint, parent], 2], color="red", solid_capstyle='round')
        for i in range(len(alignment.index1)):
            ax.plot3D(np.array([data_1[index, alignment.index1[i], 0], data_2[index, alignment.index2[i], 0]]), np.array([data_1[index, alignment.index1[i], 1], data_2[index, alignment.index2[i], 1]]), np.array([data_1[index, alignment.index1[i], 2], data_2[index, alignment.index2[i], 2]]), color="gray", solid_capstyle='round', alpha=0.3)
        
        
        
        # plt.axis('off')
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        

    ani = FuncAnimation(fig, update_1, frames=frame_number_1, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()