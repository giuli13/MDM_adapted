import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import math

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.truebones.utils.get_opt import get_opt
from data_loaders.truebones.utils.motion_process import full_rep_to_compact_rep, compact_rep_to_full_rep, remove_joints_augmentation, add_joint_augmentation
from data_loaders.truebones.utils.motion_process import truebones_tensor_to_hml_rep, mix_joints, hml_vec2mat
from data_loaders.truebones.utils.motion_process import OBJECT_SUBSETS_DICT #, MAMMALS_NO_CAT, MAMMALS_NO_SANDMOUSE, MAMMALS_NO_COMODOA, INSECTS_NO_CRAB
from utils.parents_traverse import get_inv_perm
from model.conditioners import T5Conditioner



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

""" extract parents based on first frame """
def get_motion_parents(motion):
    joints_num = motion.shape[1]
    parents_map = np.sum(motion[0]**2, axis=2)
    parents = [-1]
    for j in range(1, joints_num):
        j_parent = np.where(parents_map[j] != 0)[0][0]
        parents.append(j_parent)
    return parents

""" create temporal mask template for window size"""
def create_temporal_mask_for_window(window, max_len):
    margin = window // 2
    mask = torch.zeros(max_len+1, max_len+1)
    mask[:, 0] = 1
    for i in range(max_len+1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2Motion2ParentsDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, temporal_window_size = 31):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.max_joints = 143
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if len(motion) < min_motion_len or len(motion) >= 200:
                    continue
                text_data = []
                parents = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        parents = [int(e) for e in line_split[6].strip('][').split(', ')]
                        joints_perm = [int(e) for e in line_split[7].strip('][').split(', ')]
                        object_type = line_split[4]
                        if object_type == 'Human':
                            fname = object_type+"_kinematic_mask_full.npy"
                        else:
                            fname = object_type+".npy"

                        adj_mask = np.load(pjoin(opt.adj_masks_dir, object_type+".npy"))
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                       'parents': parents,
                                       'object_type' : object_type,
                                       'joints_perm': joints_perm,
                                       'adj_mask' : adj_mask}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window_size, self.max_motion_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data, object_type, joints_perm):
        # data [bs, njoints, nfeats, nframes], bs = 1 
        # self.mean[obj/std] (frames, joints, joints, feature_len)
        n_joints = len(joints_perm)
        data = data[0, :, :n_joints, :] # (bs, frames, joints, feature_len)
        mean_compact_permuted = self.mean[object_type][joints_perm]
        std_compact_permuted = self.std[object_type][joints_perm] # not necessary
        output = data * std_compact_permuted + mean_compact_permuted
        return output.numpy()
    
    def inv_transform_batch(self, data, object_type):
        mean_compact = self.mean[object_type][None,:]
        std_compact = self.std[object_type][None,:] 
        motions = data * std_compact + mean_compact 
        return motions.numpy()
    
    def mat2vec(self, data, object_type, n_joints):
        # assuming all motion are generated with original joints order
        assert object_type == 'Human' #for human evaluation only !
        data = data[:, :, :n_joints, :] # (bs, frames, joints, feature_len)
        hml_rep_data = truebones_tensor_to_hml_rep(data)
        return hml_rep_data 

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, parents= data['motion'], data['length'], data['text'], data['parents']
        object_type = data['object_type']
        joints_perm = data['joints_perm']
        adj_mask = data['adj_mask']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # # Crop the motions in to times of 4, and introduce small variations
        # if self.opt.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        # Crop motion if len(motion) > self.max_motion_length 
        if self.max_motion_length < len(motion):
            m_length = self.max_motion_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        mean_permuted_compact_rep = self.mean[object_type][joints_perm]
        std_permuted_compact_rep = self.std[object_type][joints_perm]
        motion_compact, _ = full_rep_to_compact_rep(motion)
        motion_compact = (motion_compact - mean_permuted_compact_rep) / std_permuted_compact_rep
        motion = compact_rep_to_full_rep(motion_compact, parents)
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1],  motion.shape[2],  motion.shape[3]))
                                     ], axis=0)
        adj_mask = adj_mask[joints_perm][:, joints_perm] # apply perm on rows and than on cols


        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), parents, self.max_joints, self.temporal_mask_template, object_type, adj_mask


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 120
        self.pointer = 0


        data_dict = {}
        id_list = [f[:-4] for f in os.listdir(opt.motion_dir)]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data, object_type):
        return data * self.std[object_type] + self.mean[object_type]
    
    def norm_obj(self, data, object_type):
        return (data - self.mean[object_type]) / self.std[object_type] 

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.max_length, None
    

# A wrapper class for t2m original dataset for MDM purposes
class Truebones(data.Dataset):
    def __init__(self, mode, datapath='./dataset/truebones_opt.txt', split="train", **kwargs):
        self.mode = mode
        
        self.dataset_name = 'truebones'
        self.dataname = 'truebones'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        self.max_joints = opt.max_joints
        self.feature_len = opt.feature_len
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = dict()
            self.std = dict()
            for file_name in os.listdir(pjoin(opt.data_root, "mean")):
                object_type = file_name[:-9]
                obj_mean = np.load(pjoin(opt.data_root, "mean", f'{object_type}_Mean.npy'))
                obj_std = np.load(pjoin(opt.data_root, "std", f'{object_type}_Std.npy'))
                self.mean[object_type] = obj_mean
                self.std[object_type] = obj_std

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2Motion2ParentsDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return len(self.t2m_dataset)



'''For use of training text motion matching model, and evaluations'''
class Text2MotionMixedJointsDataset(data.Dataset):
    def __init__(self, opt, cond_dict, split_file, w_vectorizer, temporal_window, t5_name, balanced, tpos_not_normalized):
        print("in Text2MotionMixedJointsDataset constructor")
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.cond_dict = cond_dict
        self.balanced = balanced
        self.tpos_not_normalized = tpos_not_normalized
        data_dict = {}
        all_object_types = self.cond_dict.keys()
        new_name_list = []
        length_list = []
        self.t5_conditioner = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')

        for object_type in all_object_types:
            parents = self.cond_dict[object_type]['parents']
            tpos_first_frame = self.cond_dict[object_type]['tpos_first_frame']
            joint_relations = self.cond_dict[object_type]['joint_relations']
            joints_graph_dist = self.cond_dict[object_type]['joints_graph_dist']
            normalized_edge_len = self.cond_dict[object_type]['normalized_edge_len']
            offsets = self.cond_dict[object_type]['offsets']
            joints_names = self.cond_dict[object_type]['joints_names']
            joints_names_embs = self.encode_joints_names(joints_names).detach().cpu().numpy()
            # embed joints names
            kinematic_chains = self.cond_dict[object_type]['kinematic_chains']
            if object_type == 'Human':
                object_motions = []
                with cs.open(split_file, 'r') as f:
                    for line in f.readlines():
                        object_motions.append(line.strip())
            else:
                object_motions = [f for f in os.listdir(opt.motion_dir) if f.startswith(f'{object_type}_')]
            
            for name in object_motions:
                try:
                    if object_type == 'Human':
                        motion = hml_vec2mat(np.load(pjoin(opt.motion_dir, name + '.npy')))
                    else:
                        motion = np.load(pjoin(opt.motion_dir, name))
                    data_dict[name] = {
                                        'motion': motion,
                                        'length': len(motion),
                                        'object_type': object_type,
                                        'parents': parents,
                                        'joints_graph_dist': joints_graph_dist,
                                        'joints_relations': joint_relations,
                                        'normalized_edge_len': normalized_edge_len,
                                        'tpos_first_frame': tpos_first_frame,
                                        'offsets': offsets,
                                        'joints_names_embs': joints_names_embs,
                                        'kinematic_chains': kinematic_chains
                                       }
                                       
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except:
                    pass
                
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.temporal_mask_template = create_temporal_mask_for_window(temporal_window, self.max_motion_length)
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def encode_joints_names(self, joints_names): # joints names should be padded with None to be of max_len 
        names_tokens = self.t5_conditioner.tokenize(joints_names)
        embs = self.t5_conditioner(names_tokens)
        return embs
    
    def inv_transform(self, x, y):
        mean = self.cond_dict[y['object_type']]['mean']
        std = self.cond_dict[y['object_type']]['std']
        return x * std + mean
    
    def augment(self, data):
        object_type = data['object_type']
        if object_type != "Dragon":
            aug_type = random.choice([0, 1, 2])
        else: 
            aug_type = random.choice([0, 1])
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        std[std == 0] = 1.0 # avoid division by 0 
        if aug_type == 0: #no augmentation
            return data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['normalized_edge_len'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains'], mean, std
        elif aug_type == 1: # remove_joints
            removal_rate = random.choice([0.1, 0.2, 0.3])
            return  remove_joints_augmentation(data, removal_rate, mean, std) 
        else: #add joint
            return add_joint_augmentation(data, mean, std)
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        if self.balanced:
            idx = item #self.pointer + item (handled in weighted sampler)
        else:
            idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        # motion, m_length, object_type, parents, joints_graph_dist, joints_relations, normalized_edge_len, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std = self.augment(data)
        (# disable augmantations
            motion,
            m_length,
            object_type,
            parents,
            joints_graph_dist,
            joints_relations,
            normalized_edge_len,
            tpos_first_frame,
            offsets,
            joints_names_embs
        ) = (
            data['motion'],
            data['length'],
            data['object_type'],
            data['parents'],
            data['joints_graph_dist'],
            data['joints_relations'],
            data['normalized_edge_len'],
            data['tpos_first_frame'],
            data['offsets'],
            data['joints_names_embs']
        )
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        std[std == 0] = 1.0 # avoid division by 0 

        "Z Normalization"
        # Normalize all coords but rotations 
        motion = (motion - mean[None, :]) / std[None, :]
        if not self.tpos_not_normalized:
            tpos_first_frame = (tpos_first_frame - mean) / std
            tpos_first_frame = np.nan_to_num(tpos_first_frame)
        motion = np.nan_to_num(motion)
        ind = 0
        
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        elif m_length > self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length]
            m_length = self.max_motion_length
            

        return motion, m_length, parents, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, normalized_edge_len, object_type, joints_names_embs, ind, mean, std

class TruebonesSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        num_samples = len(data_source)
        object_types = data_source.t2m_dataset.cond_dict.keys()
        name_list = data_source.t2m_dataset.name_list
        total_samples = len(name_list)
        weights = np.zeros(total_samples)
        object_share = 1.0/len(object_types)
        pointer = data_source.t2m_dataset.pointer
        for object_type in object_types:
            if object_type == "Human":
                object_indices = [i for i in range(num_samples) if i>=pointer]
            else:
                object_indices = [i for i in range(num_samples) if i>=pointer and name_list[i].startswith(f'{object_type}_')]
            object_prob = object_share / len(object_indices)
            weights[object_indices] = object_prob
        super().__init__(num_samples=num_samples, weights=weights)
    

class TruebonesMixedJoints(data.Dataset):
    def __init__(self, mode, datapath='./dataset/truebones_opt.txt', split="train", debug=False, temporal_window=31, hml=False, t5_name='t5-base', **kwargs):
        if debug and split == 'train':
            split="test"
        print("in TruebonesMixedJoints constructor")
        self.mode = mode
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath,)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path,  device, hml)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.tpos_not_normalized = kwargs['tpos_not_normalized']
        self.objects_subset = kwargs['objects_subset']
        print('Loading dataset %s ...' % opt.dataset_name)
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        subset = OBJECT_SUBSETS_DICT[self.objects_subset] # MAMMALS_NO_COMODOA 
        cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
        print(f'Dataset subset {self.objects_subset} consists of {len(cond_dict.keys())} characters')
            
        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        
        if mode == 'text_only': # NotImplemented
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionMixedJointsDataset(self.opt, cond_dict, self.split_file, self.w_vectorizer, temporal_window, t5_name, self.balanced, self.tpos_not_normalized)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
