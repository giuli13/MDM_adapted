from argparse import ArgumentParser
import argparse
import os
import json
import copy


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]
    
    # load args from model
    assert not isinstance(args, list) and not isinstance(args.model_path, list), 'Deprecated feature..'
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
        
    # backward compatibility
    if isinstance(args.emb_trans_dec, bool):
        if args.emb_trans_dec:
            args.emb_trans_dec = 'cls_tcond_cross_tcond'
        else: 
            args.emb_trans_dec = 'cls_none_cross_tcond'
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_fs", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_ric_pos", default=0.0, type=float, help="ric positions loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    # group.add_argument("--per_head_spatial_pe", action='store_true',
    #                    help="spatial pe type")
    group.add_argument("--masked_attn_from_step", default=-1, type=int, help="Adjucency masked attention from specified diffusion step. -1 if no masking is required.")
    group.add_argument("--t5_name", default='t5-base', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"], type=str,
                       help="Choose t5 pretrained model")
    group.add_argument("--temporal_window", default=31, type=int,
                       help="temporal window size")
    group.add_argument("--skip_t5", action='store_true',
                       help="If passed, joints names wont be added to features")
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")
    group.add_argument("--concat_tpos_first_frame", action='store_true',
                        help="If passed, will concatenate tpos first frame to the input motion in Baseline model.")
    group.add_argument("--linear_layer_per_object_type", action='store_true',
                        help="If passed, will create a linear layer per object type in Baseline model.")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'truebones', 'humanml_mat'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--tpos_not_normalized", action='store_true',
                       help="Do not normalize T-pose")
    group.add_argument("--objects_subset", default='all', choices=['all', 'mammals', 'connected_to_ground', 'flying', 'dinosaurs', 'insects', 'insects_snakes', 'mammals_clean', 'insects_clean', 'flying_clean', 'dinosaurs_clean', 'all_clean', "mammals_no_sand_mouse", "mammals_no_cat", "mammals_no_comodoa", "insects_snakes_no_crab"], type=str,
                       help="Object subset.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='WandBPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=5e-5, type=float, help="Learning rate.")

    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=120, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--hml", action='store_true',
                       help="train on hml data only")
    group.add_argument("--balanced", action='store_true',
                       help="Use balancing sampler for fairness between topologies")




def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--object_type", default=['Flamingo'], type=str, nargs='+',
                       help="An object type to be generated. If empty, will generate flamingo :).")
    group.add_argument("--parents", default='[]', type=str,
                       help="parents topology as condition")
    group.add_argument("--joints_perm", default='[]', type=str,
                       help="joints_perm topology as condition")
    
def add_dift_options(parser):
    # bvhs_dir, sample_bvh, face_joints, save_dir=None, tpos_bvh=None
    group = parser.add_argument_group('dift')
    group.add_argument("--apply_pca", action='store_true',
                       help="apply pca on feats before calculating similarity.")
    group.add_argument("--seen_ref", action='store_true',
                       help="ref topology is seen.")

    group.add_argument("--seen_tgt", action='store_true',
                       help="tgt topology is seen.")    
    
    group.add_argument("--bvhs_dir_ref", default='', type=str,
                       help="bvhs dir ref.")
    group.add_argument("--sample_ref", default='/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed/motions/Ostrich___Attack_548.npy', type=str,
                       help="sample bvh ref.")
    group.add_argument("--object_type_ref", default='Ostrich', type=str,
                       help="object type ref.")
    group.add_argument("--tpos_bvh_ref", default='', type=str,
                       help="tpos bvh ref.")
    group.add_argument("--face_joints_ref", default=[1, 2, 3, 4], type=int, nargs='+',
                       help="face joints of reference sample.")
    group.add_argument("--bvhs_dir_tgt", default='', type=str,
                       help="bvhs dir tgt.")
    group.add_argument("--sample_tgt", default='/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed/motions/Dragon___Fly_272.npy', type=str,
                       help="sample bvh tgt.")
    group.add_argument("--object_type_tgt", default='Dragon', type=str,
                       help="object type tgt.")
    group.add_argument("--tpos_bvh_tgt", default='', type=str,
                       help="tpos bvh tgt.")
    group.add_argument("--face_joints_tgt", default=[1, 2, 3, 4], type=int, nargs='+',
                       help="face joints of target sample.")
    group.add_argument("--tmp_save_dir", default='', type=str,
                       help="temporal save dir.")
    group.add_argument("--dift_type", default='spatial', choices=['spatial', 'temporal'], type=str,
                       help="apply dift on spatial or temporal features")

def add_pca_options(parser):
    # bvhs_dir, sample_bvh, face_joints, save_dir=None, tpos_bvh=None
    group = parser.add_argument_group('pca')
    group.add_argument("--seen", action='store_true',
                       help="topology is seen.")
    group.add_argument("--bvhs_dir", default='', type=str,
                       help="bvhs dir.")
    group.add_argument("--sample", default='/home/dcor/inbargat1/multi-skeleton-mdm/dataset/truebones/zoo/processed/motions/Ostrich___Attack_548.npy', type=str,
                       help="sample bvh/npy (depends if seen or unseen).")
    group.add_argument("--object_type", default='Ostrich', type=str,
                       help="object type.")
    group.add_argument("--tpos_bvh", default='', type=str,
                       help="tpos bvh.")
    group.add_argument("--face_joints", default=[1, 2, 3, 4], type=int, nargs='+',
                       help="face joints of sample.")
    group.add_argument("--tmp_save_dir", default='', type=str,
                       help="temporal save dir.")
    group.add_argument("--k", default=5, type=int,
                       help="Number of centroids for k-means centroids")
    
def add_generate_new_topology_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--object_type", default='Flamingo', type=str,
                       help="An object type to be generated. If empty, will generate flamingo :).")
    group.add_argument("--parents", default='[]', type=str,
                       help="parents topology as condition")
    group.add_argument("--example_bvh_path", default='', type=str,
                       help="example bvh to load parents from")
                

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

def add_ata_evaluation_options(parser):
        group = parser.add_argument_group('ata_eval')
        group.add_argument("--eval_mode", required=True, type=str, choices=['bvh', 'npy_rot', 'npy_loc'], help="Path to gt dir.")
        group.add_argument("--benchmark_path", default='ata_eval/benchmark_names.txt', type=str,  help="Path to benchmark character names. If empty, will use all excluding the characters_to_exclude")
        group.add_argument("--eval_gt_dir", required=True, type=str, help="Path to gt dir.")
        group.add_argument("--eval_gen_dir", required=True, type=str, help="Path to gen dir.")
        group.add_argument("--characters_to_exclude", default='MouseyNoFingers,Mousey_m,Trex,SabreToothTiger,Raptor2', type=str, help="Comma separated list of characters to exclude. The default is character with more than 40 motions.")
        group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")


def add_ata_evaluation_stats_options(parser):
        group = parser.add_argument_group('ata_eval_stats')
        group.add_argument("--eval_mode", required=True, type=str, choices=['bvh', 'npy_rot', 'npy_loc', 'npy_relative_loc'], help="Path to gt dir.")
        group.add_argument("--benchmark_path", default='ata_eval/benchmark_names.txt', type=str,  help="Path to benchmark character names. If empty, will use all excluding the characters_to_exclude")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    elif args.dataset in ['truebones', 'humanml_mat']:
        cond_mode = 'object_type'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def dift_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_dift_options(parser)
    args = parse_and_load_from_model(parser)

    return args

def pca_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_pca_options(parser)
    args = parse_and_load_from_model(parser)

    return args

def generate_new_topology_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_new_topology_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)

def ata_evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_ata_evaluation_options(parser)
    return parser.parse_args()

def ata_evaluation_stats_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_ata_evaluation_stats_options(parser)
    return parser.parse_args()
