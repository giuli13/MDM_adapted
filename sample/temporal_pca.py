# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import pca_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model_wo_clip
from utils import dist_util
# from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.utils.plot_script import plot_general_skeleton_pca, plot_general_skeleton_kmeans,  save_sample
from data_loaders.tensors import truebones_mixed_collate
from data_loaders.truebones.utils.motion_process import recover_from_ric_np, recover_from_rot_np, process_new_object_type
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from data_loaders.humanml.common.skeleton import Skeleton
# from sklearn.preprocessing import normalize
from model.conditioners import T5Conditioner
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def normalize(data, axis):
    return (data - data.min(dim=axis, keepdim=True).values) / (data.max(dim=axis, keepdim=True).values - data.min(dim=axis, keepdim=True).values + 1e-7)

def normalize_np(data, axis):
    return (data - data.min(axis=axis, keepdims=True)) / (data.max(axis=axis, keepdims=True) - data.min(axis=axis, keepdims=True) + 1e-7)

def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def prepare_pca_inputs(motion, object_type, cond_dict, temporal_window, t5_conditioner, tpos_not_normalized=False, rots_not_normalized=False):
    batches = list()
    batch=list()
    mean = cond_dict['mean']
    std = cond_dict['std']
    motion = (motion-mean[None]) / std[None]
    motion = np.nan_to_num(motion)
    batch.append(motion)
    batch.append(motion.shape[0])
    batch.append(cond_dict['parents'])
    if tpos_not_normalized:
        batch.append(cond_dict['tpos_first_frame'])
    else:
        tpos = (cond_dict['tpos_first_frame'] - mean) / std
        tpos = np.nan_to_num(tpos)
        batch.append(tpos)
    batch.append(cond_dict['offsets'])
    batch.append(create_temporal_mask_for_window(temporal_window, motion.shape[0]))
    batch.append(cond_dict['joints_graph_dist'])
    batch.append(cond_dict['joint_relations'])
    batch.append(cond_dict['normalized_edge_len'])
    batch.append(object_type)
    batch.append(encode_joints_names(cond_dict['joints_names'], t5_conditioner).detach().cpu().numpy())
    batch.append(0)
    batch.append(mean)
    batch.append(std)
    batches.append(batch)
    return truebones_mixed_collate(batches)


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
        motions_dir = "/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_bvh_rots/motions"
        object_cond_dict = cond[object_type]
        if motion_path is None:
            all_motions = [pjoin(motions_dir, f) for f in os.listdir(motions_dir) if f.endswith('.bvh') and f.startswith(f'{object_type}__')]
            motion_path = np.load(random.choice(all_motions))
        motion = np.load(motion_path)
        return motion, object_cond_dict
    else:
        return process_new_object_type(bvhs_dir, motion_path, face_joints, save_dir=save_dir, tpos_bvh=tpos_bvh)
    
def vis_kmeans(t, layer, activations, object_cond, motion, model_path, n_clusters=5):
    ''' Prepare features for visualization '''
    relevant_activations = activations[t][layer][:, 0]
    motion_len = motion.shape[0]
    n_joints = motion.shape[1]
    parents = object_cond["parents"]
    relevant_activations = relevant_activations[1:motion_len+1, :n_joints].transpose(0, 1)
    # relevant_activations = relevant_activations.mean(0)
    # normalized_activations = normalize(relevant_activations, axis=1).detach().cpu().numpy()
    normalized_activations = relevant_activations.detach().cpu().numpy()
    normalized_activations = normalized_activations.mean(0)
    n_components = 3
    pca = PCA(n_components=n_components)
    feat_pca = pca.fit_transform(normalized_activations)
    kmeans = KMeans(n_clusters=n_clusters)
    # kmeans.fit(normalized_activations)
    kmeans.fit(feat_pca)
    positions, _ = recover_from_ric_np(motion[:motion_len], parents)
    ani = plot_general_skeleton_kmeans(parents, kmeans.labels_, n_clusters, positions, "", "truebones", fps=20)
    # fname = f"diffusion_step_{t}_layer_{layer}_{object_cond['object_type']}_root.mp4"
    fname = f"diffusion_step_{t}_layer_{layer}_{object_cond['object_type']}_pca.mp4"
    model_dir =  os.path.dirname(model_path)
    out_dir = pjoin(model_dir, f"kmean_out_{os.path.basename(model_path)[:-3]}")
    os.makedirs(out_dir, exist_ok=True)
    save_sample(out_dir, fname, ani, 20, motion_len)

def vis_pca(t, layer, activations, object_cond, motion, model_path):
    # color per kinematic chain, not vertex 
    # plot_single_frame_kinchains
    # [ref, tgt]
    relevant_activations = activations[t][layer][:, 0]
    motion_len = motion.shape[0]
    n_joints = motion.shape[1]
    parents = object_cond["parents"]
    relevant_activations = relevant_activations[1:motion_len+1, :n_joints].transpose(0, 1)
    relevant_activations = relevant_activations.mean(0)
    normalized_activations = normalize(relevant_activations, axis=0).detach().cpu().numpy()
    n_components = 3
    pca = PCA(n_components=n_components)
    feat_pca = pca.fit_transform(normalized_activations)
    #strech in the range 0-1
    feat_pca = normalize_np(feat_pca, axis=0)
    positions, _ = recover_from_ric_np(motion[:motion_len], parents)
    ani = plot_general_skeleton_pca(parents, feat_pca, positions, "", "truebones", fps=20)
    fname = f"diffusion_step_{t}_layer_{layer}_{object_cond['object_type']}.mp4"
    model_dir =  os.path.dirname(model_path)
    out_dir = pjoin(model_dir, "pca_out")
    os.makedirs(out_dir, exist_ok=True)
    save_sample(out_dir, fname, ani, 20, motion_len)
    
    
    

def run_pca(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = pca_args()
    fixseed(args.seed)    
    cond_dict = np.load("/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed_bvh_rots/cond.npy", allow_pickle=True).item()
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dist_util.setup_dist(args.device)
    motion, object_cond_dict = process_object_type(seen=args.seen, motion_path=args.sample, object_type=args.object_type, cond=cond_dict)


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
    motions, model_kwargs = prepare_pca_inputs(motion, args.object_type, object_cond_dict, args.temporal_window, t5_conditioner)


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
        
        for t in [0, 1, 2 ,3, 4,5, 6,7, 8,9, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for layer in range(args.layers):
                vis_kmeans(t, layer, activations, object_cond_dict, motion, args.model_path)
    

if __name__ == "__main__":
    run_pca()