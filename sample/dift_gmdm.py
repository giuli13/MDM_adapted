# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import dift_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model_wo_clip
from utils import dist_util
# from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.utils.plot_script import plot_general_skeleton_3d_motion, plot_general_skeleton_correspondance, save_multiple_samples
from data_loaders.tensors import truebones_mixed_collate
from data_loaders.truebones.utils.motion_process import recover_from_ric_np, recover_from_rot_np, process_new_object_type
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from data_loaders.humanml.common.skeleton import Skeleton
from sklearn.preprocessing import normalize
from model.conditioners import T5Conditioner
import random


def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def prepare_dift_inputs(motion_ref, object_type_ref, cond_dict_ref, motion_tgt, object_type_tgt,cond_dict_tgt, temporal_window, t5_conditioner, tpos_not_normalized=False, rots_not_normalized=False):
    batches = list()
    batch_ref=list()
    mean_ref = cond_dict_ref['mean']
    std_ref = cond_dict_ref['std']
    mean_tgt= cond_dict_tgt['mean']
    std_tgt = cond_dict_tgt['std']
    motion_ref = (motion_ref - mean_ref[None]) / std_ref[None]
    motion_ref = np.nan_to_num(motion_ref)
    batch_ref.append(motion_ref)
    batch_ref.append(motion_ref.shape[0])
    batch_ref.append(cond_dict_ref['parents'])
    # normalize tpos
    if tpos_not_normalized:
        tpos_ref = cond_dict_ref['tpos_first_frame']
    else:
        tpos_ref = (cond_dict_ref['tpos_first_frame'] - mean_ref) / std_ref
        tpos_ref = np.nan_to_num(tpos_ref)
        
    batch_ref.append(tpos_ref)
    batch_ref.append(cond_dict_ref['offsets'])
    batch_ref.append(create_temporal_mask_for_window(temporal_window, max(motion_ref.shape[0], motion_tgt.shape[0])))
    batch_ref.append(cond_dict_ref['joints_graph_dist'])
    batch_ref.append(cond_dict_ref['joint_relations'])
    batch_ref.append(cond_dict_ref['normalized_edge_len'])
    batch_ref.append(object_type_ref)
    batch_ref.append(encode_joints_names(cond_dict_ref['joints_names'], t5_conditioner).detach().cpu().numpy())
    batch_ref.append(0)
    batch_ref.append(mean_ref)
    batch_ref.append(std_ref)
    batches.append(batch_ref)
    
    batch_tgt=list()
    motion_tgt = (motion_tgt - mean_tgt[None]) / std_tgt[None]
    motion_tgt = np.nan_to_num(motion_tgt)
    batch_tgt.append(motion_tgt)
    batch_tgt.append(motion_tgt.shape[0])
    batch_tgt.append(cond_dict_tgt['parents'])
    # normalize tpos
    if tpos_not_normalized:
        tpos_tgt = cond_dict_tgt['tpos_first_frame']
    else:
        tpos_tgt = (cond_dict_tgt['tpos_first_frame'] - mean_tgt) / std_tgt
        tpos_tgt = np.nan_to_num(tpos_tgt)
        
    batch_tgt.append(tpos_tgt)
    batch_tgt.append(cond_dict_tgt['offsets'])
    batch_tgt.append(create_temporal_mask_for_window(temporal_window, max(motion_ref.shape[0], motion_tgt.shape[0])))
    batch_tgt.append(cond_dict_tgt['joints_graph_dist'])
    batch_tgt.append(cond_dict_tgt['joint_relations'])
    batch_tgt.append(cond_dict_tgt['normalized_edge_len'])
    batch_tgt.append(object_type_tgt)
    batch_tgt.append(encode_joints_names(cond_dict_tgt['joints_names'], t5_conditioner).detach().cpu().numpy())
    batch_tgt.append(0)
    batch_tgt.append(mean_tgt)
    batch_tgt.append(std_tgt)
    batches.append(batch_tgt)
        
    return truebones_mixed_collate(batches)

def create_args():
    pass

"""
Returns motion features of the given BVH file. besides bvh_path and face_joints, all other parameters 
are optional, but if supplied might improve DIFT results. 

bvh_path: path to a bvh file of a given animaiton 
face_joints: based on these joints the orientation is determined. 
             Should be given in the order [right hip, left hip, right soulder, left shoulder] 
seen_topology: if the bvh topology already seen by the model (i.e, belongs to it's train set)
object_type: relevant only if bvh is from an already seen topology. 
bvh_tpos: bvh of rest pose. relevant mostly if the given bvh tpos is unnatural 
feet: feet indices. The order does not matter 
"""
def process_object_type(seen, object_type, motion_path=None, bvhs_dir=None, face_joints=None, save_dir=None, tpos_bvh=None, cond=None):
    if seen:
        motions_dir = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_fc/motions"
        object_cond_dict = cond[object_type]
        if motion_path is None:
            all_motions = [pjoin(motions_dir, f) for f in os.listdir(motions_dir) if f.endswith('.bvh') and f.startswith(f'{object_type}__')]
            motion_path = np.load(random.choice(all_motions))
        motion = np.load(motion_path)
        return motion, object_cond_dict
    else:
        return process_new_object_type(bvhs_dir, object_type, motion_path, face_joints, save_dir=save_dir, tpos_bvh=tpos_bvh)
    

def vis_dift(t, layer, activations, cond_ref, cond_tgt, motion_ref, motion_tgt, model_path, dift_type='spatial'):
    # color per kinematic chain, not vertex 
    # plot_single_frame_kinchains
    # [ref, tgt]
    buckets = [0, 14, 22, 35, 51, 70, 84, 95, 129]
    values = [0, 1, 2, 3, 2, 3, 1, 0]
    representators = {0: 7, 1: 20, 2: 25, 3:42}
    relevant_activations = activations[t][layer]
    ref_len = motion_ref.shape[0]
    ref_n_joints = motion_ref.shape[1]
    tgt_len = motion_tgt.shape[0]
    ref_kinchains = cond_ref["kinematic_chains"]
    n_kinchains = len(ref_kinchains)
    tgt_n_joints = motion_tgt.shape[1]
    if dift_type == 'spatial':
        min_length = min(ref_len, tgt_len)
        ref_activations = relevant_activations[:min_length, 0, :ref_n_joints]
        tgt_activations = relevant_activations[:min_length, 1, :tgt_n_joints]
        
    if dift_type =='temporal':
        ref_activations = relevant_activations[:ref_len, 0, :ref_n_joints]
        tgt_activations = relevant_activations[:tgt_len, 1, :tgt_n_joints]
        ref_activations = ref_activations.transpose(0, 1)
        tgt_activations = tgt_activations.transpose(0, 1)
        
    ref_activations = ref_activations.mean(0)
    tgt_activations = tgt_activations.mean(0)
    
    if dift_type == 'temporal':
        distinct_values = set(values)
        for val in distinct_values:
            entries = list()
            indices = [i for i, x in enumerate(values) if x == val]
            for i in indices:
                entries += [l for l in range(buckets[i], buckets[i+1])]
            entries = torch.tensor(entries, device = ref_activations.device)
            ref_activations[entries] = ref_activations[representators[val]]
        
        
        
        
    cos_sim=tgt_activations @ ref_activations.transpose(-1, -2)
    print(cos_sim.shape)
    vec_norms = (torch.norm(tgt_activations, dim=-1)[..., None] @ torch.norm(ref_activations, dim=-1)[..., None].transpose(-1, -2))
    print(vec_norms.shape)
    cos_sim = cos_sim/vec_norms
    correspondance = torch.argmax(cos_sim, dim=-1)
    if dift_type == 'temporal':

        joint2color_ref = dict()
        for frame in range(ref_len):
            for b in range(len(buckets)-1):
                if frame < buckets[b+1]:
                    break
            frames_cls = values[b]
            joint2color_ref[frame] = dict()
            for j in range(ref_n_joints):
                joint2color_ref[frame][j] = frames_cls

        joint2color_tgt = dict()
        for frame in range(tgt_len):
            joint2color_tgt[frame] = dict()
            for j in range(tgt_n_joints):
                joint2color_tgt[frame][j] = joint2color_ref[correspondance[frame].item()][0]
    else:
        joint2color_ref = dict()
        for frame in range(min_length):
            joint2color_ref[frame] = dict()
            for chain_ind, chain in enumerate(ref_kinchains):
                for j in chain:
                    if len(chain) <= 5 and chain[0] != 0:
                        joint2color_ref[frame][j] = joint2color_ref[frame][cond_ref["parents"][j]]
                    else:
                        joint2color_ref[frame][j] = chain_ind
                    if j == 34:
                        joint2color_ref[frame][j] = 0
                    if j == 8:
                        joint2color_ref[frame][j] = 21
                    

        joint2color_tgt = dict()
        for frame in range(min_length):
            joint2color_tgt[frame] = dict()
            for j in range(tgt_n_joints):
                joint2color_tgt[frame][j] = joint2color_ref[frame][correspondance[j].item()]
    

    if dift_type=='temporal':
        positions_ref, _ = recover_from_ric_np(motion_ref[:ref_len], cond_ref["parents"])
        print(positions_ref.shape)
        positions_tgt, _ = recover_from_ric_np(motion_tgt[:tgt_len], cond_tgt["parents"])
        print(positions_tgt.shape)
        animations = np.empty(shape=(1, 2), dtype=object)
        animations[0, 0] = plot_general_skeleton_correspondance(cond_ref["parents"], joint2color_ref, len(set(values)), positions_ref, "", "truebones", fps=20)
        animations[0, 1] = plot_general_skeleton_correspondance(cond_tgt["parents"], joint2color_tgt, len(set(values)), positions_tgt, "", "truebones", fps=20) 
    else:
        positions_ref, _ = recover_from_ric_np(motion_ref[:min_length], cond_ref["parents"])
        print(positions_ref.shape)
        positions_tgt, _ = recover_from_ric_np(motion_tgt[:min_length], cond_tgt["parents"])
        print(positions_tgt.shape)
        animations = np.empty(shape=(1, 2), dtype=object)
        animations[0, 0] = plot_general_skeleton_correspondance(cond_ref["parents"], joint2color_ref, n_kinchains, positions_ref, "", "truebones", fps=20)
        animations[0, 1] = plot_general_skeleton_correspondance(cond_tgt["parents"], joint2color_tgt, n_kinchains, positions_tgt, "", "truebones", fps=20)
    if dift_type=='temporal':
        fname = f"AAAdiffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_temporal.mp4"
        npy_fname = f"AAAdiffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_temporal.npy"
    else:
        fname = f"diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_spatial.mp4"
        npy_fname = f"diffusion_step_{t}_layer_{layer}_ref_{cond_ref['object_type']}_tgt_{cond_tgt['object_type']}_spatial.npy"
    model_dir = os.path.dirname(model_path)
    model_number = os.path.basename(model_path)[:-3]
    
    out_dir = pjoin(model_dir, model_number, "dift_out")
    os.makedirs(out_dir, exist_ok=True)
    mapping_dict = {"ref": joint2color_ref, "tgt": joint2color_tgt}
    np.save(pjoin(out_dir, npy_fname), mapping_dict, allow_pickle=True)
    if dift_type == 'spatial':
        save_multiple_samples(out_dir, fname, animations, 20, min_length) 
    else:
        save_multiple_samples(out_dir, fname, animations, 20, max(tgt_len, ref_len)) 
        
    
    
    

def run_dift(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = dift_args()
    fixseed(args.seed)    
    cond_dict = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_fc/cond.npy", allow_pickle=True).item()
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dist_util.setup_dist(args.device)
    motion_ref, cond_dict_ref = process_object_type(seen=args.seen_ref, motion_path=args.sample_ref, object_type=args.object_type_ref, cond=cond_dict)
    motion_tgt, cond_dict_tgt = process_object_type(seen=args.seen_tgt, motion_path=args.sample_tgt, object_type=args.object_type_tgt, cond=cond_dict)


    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    # this block must be called BEFORE the dataset is loaded
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = 2  # Sampling a single batch from the testset, with exactly args.num_samples
    args.num_repetitions = 1
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
    
    # motion_ref, object_type_ref, cond_dict_ref, motion_tgt, object_type_tgt,cond_dict_tgt, temporal_window, t5_conditioner
    motions, model_kwargs = prepare_dift_inputs(motion_ref, args.object_type_ref, cond_dict_ref, motion_tgt, args.object_type_tgt, cond_dict_tgt, args.temporal_window, t5_conditioner, tpos_not_normalized=args.tpos_not_normalized)



    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample, activations = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            motions.shape,  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=motions.to(device=dist_util.dev()),
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            get_activations=True
        )
        
        for t in [0, 2, 4, 6, 7, 8, 9, 10]:
            for layer in [0, 1, 2, 3]:
                vis_dift(t, layer, activations, cond_dict_ref, cond_dict_tgt, motion_ref, motion_tgt, args.model_path, dift_type=args.dift_type)
    

if __name__ == "__main__":
    run_dift()