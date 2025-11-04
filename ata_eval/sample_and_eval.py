from argparse import ArgumentParser
from ata_eval.eval_truebones import eval_full_benchmark
from utils.parser_util import add_base_options, add_ata_evaluation_options, add_data_options, add_sampling_options, parse_and_load_from_model
from utils.model_util import create_model_and_diffusion_baseline, load_model_wo_clip
import torch
from utils.fixseed import fixseed
import os
import numpy as np
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from sample.generate_baseline import create_condition
import itertools
from data_loaders.truebones.utils.motion_process import recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.humanml.utils.plot_script import plot_general_skeleton_3d_motion
from os.path import join as pjoin
import multiprocessing
import BVH
import InverseKinematics
from functools import partial
import pickle
from collections import Counter


def parser():
    parser = ArgumentParser()
    add_base_options(parser) # disable random masking
    add_ata_evaluation_options(parser)
    add_sampling_options(parser)
    add_data_options(parser)
    group = parser.add_argument_group('sample_eval')
    group.add_argument('--cond_dict_path', type=str, default='dataset/truebones/zoo/processed_bvh_rots/cond.npy')
    group.add_argument('--n_frames', type=int, default=120,
                        help='Number of frames to generate for each sample')
    group.add_argument("--object_type", default=['Alligator','Anaconda','Ant','Bat','Bear','Bird','BrownBear','Buffalo','Camel','Deer','Dragon','Elephant','Fox','FireAnt','Horse','Hound','Monkey','Raptor'], type=str, nargs='+',
                       help="An object type to be generated.")
    group.add_argument('--num_gen_bvh_per_obj_type', type=int, default=20,
                        help='Number of generated BVHs per object type')
    group.add_argument('--num_parallel', type=int, default=8,
                        help='Number of parallel processes to generate BVHs')
    group.add_argument('--generate', action='store_true',
                        help='If set, generate samples')
    group.add_argument('--process', action='store_true',
                        help='If set, process the generated samples')
    group.add_argument('--eval', action='store_true',
                        help='If set, evaluate the generated samples')
    
    return parse_and_load_from_model(parser)

def generate_batch(object_types, model, diffusion, cond_dict, n_frames, temporal_window, tpos_not_normalized):
    _, model_kwargs = create_condition(object_types, cond_dict, n_frames, temporal_window, tpos_not_normalized)

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(len(object_types), device=dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop

    sample_batch = sample_fn(
        model,
        (len(object_types), model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    return sample_batch, model_kwargs

def load_model(args, cond_dict):
    model, diffusion = create_model_and_diffusion_baseline(args, cond_dict)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model) 
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion

def create_samples(args) -> list[str]:
    cond_dict = np.load(args.cond_dict_path, allow_pickle=True).item()
    n_frames = args.n_frames
    object_types = args.object_type
    if object_types[0] == 'All':
        object_types = list(set(cond_dict.keys()) - set(args.characters_to_exclude.split(',')))
    num_gen_bvh_per_obj_type = args.num_gen_bvh_per_obj_type
    os.makedirs(args.eval_gen_dir, exist_ok=True)
    out_path = pjoin(args.eval_gen_dir, 'samples')
    os.makedirs(out_path, exist_ok=True)    
    all_sample_files = []
    sample_idx_counter = Counter()

    model, diffusion = load_model(args, cond_dict)
    for batched_object_types in batched_iter(
            itertools.chain.from_iterable(
                itertools.repeat(o, num_gen_bvh_per_obj_type)
                for o in object_types
            ),
            args.batch_size
        ):
        
        sample_batch, model_kwargs = generate_batch(
            object_types=batched_object_types,
            model=model,
            diffusion=diffusion,
            cond_dict=cond_dict,
            n_frames=n_frames,
            temporal_window=args.temporal_window,
            tpos_not_normalized=args.tpos_not_normalized
            )
        
        bs, max_joints, n_feats, n_frames = sample_batch.shape
        for i, motion in enumerate(sample_batch):
            n_joints = model_kwargs['y']["n_joints"][i].item()
            motion = motion[:n_joints]
            object_type = model_kwargs['y']["object_type"][i]
            parents = model_kwargs['y']["parents"][i]
            mean = cond_dict[object_type]['mean'][None, :]
            std = cond_dict[object_type]['std'][None, :]
            motion = motion.permute(2, 0, 1).cpu().numpy() * std + mean

            # sample_dict = {
            #     'motion': motion,
            #     'object_type': object_type,
            #     'parents': parents,
            # }
            sample_file = pjoin(out_path, f"{object_type}_{sample_idx_counter[object_type]}.npy")
            print(f"Saving sample to [{sample_file}]")
            np.save(sample_file, motion)
            all_sample_files.append(sample_file)
            sample_idx_counter[object_type] += 1
    return all_sample_files

def process_sample(sample_file: str, eval_gen_dir, cond_dict_path):
    motion = np.load(sample_file)
    cond_dict = np.load(cond_dict_path, allow_pickle=True).item()
    object_type = os.path.basename(sample_file).split('_')[0]
    parents = cond_dict[object_type]['parents']
    filename = os.path.basename(sample_file)
    base_name = os.path.splitext(filename)[0]
    print(f"Processing sample [{filename}] of type [{object_type}]")
    global_positions, _ = recover_from_bvh_ric_np(motion, parents)
    bvh_name = f"{base_name}.bvh"
    mp4_name = f"{base_name}.mp4"
    os.makedirs(pjoin(eval_gen_dir, 'bvhs'), exist_ok=True)
    bvh_path = pjoin(eval_gen_dir, 'bvhs', bvh_name)
    os.makedirs(pjoin(eval_gen_dir, 'animation'), exist_ok=True)
    mp4_path = pjoin(eval_gen_dir, 'animation', mp4_name)
    joints_names = cond_dict[object_type]['joints_names']
    offsets = cond_dict[object_type]['offsets']
    plot_general_skeleton_3d_motion(mp4_path, parents, global_positions, dataset='truebones', title=base_name, fps=20)
    anim, sorted_order, parents = InverseKinematics.animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
    BVH.save(bvh_path, anim, [joints_names[i] for i in sorted_order])

def parallel_process_all_samples(args):
    object_types = args.object_type
    samples_dir = pjoin(args.eval_gen_dir, 'samples')
    sample_files = os.listdir(samples_dir)
    sample_files = filter(lambda x: x.endswith('.npy'), sample_files)
    sample_files = filter(lambda x: x.split('_')[0] in object_types, sample_files)
    sample_files = map(lambda x: pjoin(samples_dir, x), sample_files)

    with multiprocessing.get_context("spawn").Pool(args.num_parallel) as pool:
        pool.map(
            partial(process_sample,
                    eval_gen_dir=args.eval_gen_dir,
                    cond_dict_path=args.cond_dict_path
                    ),
            sample_files
        )
    

def batched_iter(iterable, n):
    # batched_iter('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


if __name__ == "__main__":
    args = parser()
    fixseed(args.seed)
    assert os.path.isdir(args.eval_gt_dir), f'Invalid gt dir [{args.eval_gt_dir}]'
    if args.generate and args.eval and not args.process:
        raise ValueError('Cannot generate and evaluate without processing the generated samples')
    
    if args.generate:
        all_sample_files = create_samples(args)
    if args.process:
        parallel_process_all_samples(args)
    if args.eval:
        log_file = os.path.join(os.path.dirname(args.eval_gen_dir),
                                'eval_' + os.path.basename(args.eval_gen_dir) + '.log')
        print(f'Will save to log file [{log_file}]')
        eval_dict = eval_full_benchmark(args)
        with open(log_file, 'w') as fw:
            fw.write(str(eval_dict))
        np.save(log_file.replace('.log', '.npy'), eval_dict)