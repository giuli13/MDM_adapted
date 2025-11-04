# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
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
from data_loaders.truebones.utils.motion_process import OBJECT_SUBSETS_DICT
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from data_loaders.humanml.common.skeleton import Skeleton
from sklearn.preprocessing import normalize
from model.conditioners import T5Conditioner
import BVH
import math 
from InverseKinematics import animation_from_positions

def main(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    
    if cond_dict is None:
        if args.object_type[0] =='Human':
            cond_dict = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed/hml_cond.npy", allow_pickle=True).item()
        else:
            cond_dict = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_bvh_rots/cond.npy", allow_pickle=True).item()
            subset = OBJECT_SUBSETS_DICT[args.objects_subset]
            cond_dict = {k:cond_dict[k] for k in subset if k in cond_dict}
            print(f'Dataset subset {args.objects_subset} consists of {len(cond_dict.keys())} characters')
            

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['truebones'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = 120#min(max_frames, int(args.motion_length*fps))
    max_joints = 143 if args.dataset == 'truebones' else 22
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'eval_samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(pjoin(out_path, "bvh"), exist_ok=True)
    os.makedirs(pjoin(out_path, "mp4"), exist_ok=True)
    os.makedirs(pjoin(out_path, "npy"), exist_ok=True)
    # this block must be called BEFORE the dataset is loaded
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    # args.batch_size = args.eval_batch  # Sampling a single batch from the testset, with exactly args.num_samples
    args.num_repetitions = 30
    # total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    # if args.guidance_param != 1:
    #     model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_object_types = list(cond_dict.keys())
    it = math.ceil(len(all_object_types) / args.batch_size)
    print(f'{it} batches are required')
    for i in range(it):
        batch_object_types = all_object_types[args.batch_size * i: args.batch_size * (i+1)]
        _, model_kwargs = create_condition(batch_object_types, cond_dict, n_frames, args.temporal_window, t5_conditioner=t5_conditioner, tpos_not_normalized = args.tpos_not_normalized)
        
        bs = len(batch_object_types)
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(bs, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (bs, max_joints, model.feature_len, n_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            bs, max_joints, n_feats, n_frames = sample.shape
            for i, motion in enumerate(sample):
                n_joints = model_kwargs['y']["n_joints"][i].item()
                motion = motion[:n_joints]
                object_type = model_kwargs['y']["object_type"][i]
                parents = model_kwargs['y']["parents"][i]
                mean = cond_dict[object_type]['mean'][None, :]
                std = cond_dict[object_type]['std'][None, :]
                offsets = cond_dict[object_type]['offsets']
                out_anim = None
                motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
                #global_positions, _ = recover_from_bvh_ric_np(motion, parents)
                #out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
                
                #global_positions, out_anim = recover_from_bvh_rot_np(motion, parents, offsets)
            
                name_pref = '%s__%d'%(object_type, rep_i)
                npy_name = f'{name_pref}.npy'
                mp4_name = f'{name_pref}.mp4'
                bvh_name = f'{name_pref}.bvh'    
                #plot_general_skeleton_3d_motion(pjoin(out_path, "mp4",  mp4_name), parents, global_positions, dataset=args.dataset, title="", fps=fps)
                np.save(pjoin(out_path, "npy" ,npy_name), motion)
                if out_anim is not None:
                    BVH.save(pjoin(out_path, "bvh", bvh_name), out_anim, cond_dict[object_type]['joints_names'])
                print("repetition #" + str(rep_i) + " ,created motion: "+ npy_name)


def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def create_condition(object_types, cond_dict, n_frames, temporal_window, t5_conditioner, tpos_not_normalized=False):
    batches = list()
    for object_type in object_types:
        batch=list()
         # motion, m_length, parents, joints_perm, inv_joints_perm, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, normalized_edge_len, object_type, joints_names
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        std[std==0.] = 1.
        # Tpos normalization if needed
        if tpos_not_normalized:
            tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        else:
            normalized_tpos = (cond_dict[object_type]['tpos_first_frame'] - mean) / std
            tpos_first_frame = np.nan_to_num(normalized_tpos)
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        normalized_edge_len = cond_dict[object_type]['normalized_edge_len']
        offsets = cond_dict[object_type]['offsets']
        #joints_names = ['Hips', 'pelvis0', 'spine', 'spine0', 'spine1', 'spine2', 'spine3', 'LUpperArm', 'LForeArm', 'LHand', 'LFinger', 'RUpperArm', 'RForeArm', 'RHand', 'RFinger', 'Neck', 'Head', 'Mouth', 'LEar', 'REar', 'LLegBack1', 'LLegBack2', 'LLegBack3', 'LFootBack', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
        joints_names_embs = encode_joints_names(cond_dict[object_type]['joints_names'] , t5_conditioner).detach().cpu().numpy()
        batch.append(np.zeros((n_frames, n_joints, 13)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(tpos_first_frame)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(normalized_edge_len)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batches.append(batch)
        
    return truebones_mixed_collate(batches)


if __name__ == "__main__":
    main()
