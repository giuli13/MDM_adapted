from model.mdm import MDM
from model.mdm_baseline import MDMBaseline
from model.mdm_multi_skeleton import GMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])
    if not all([k.startswith('clip_model.') for k in missing_keys]):
        print('Missing keys:', [k for k in missing_keys if not k.startswith('clip_model.')])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_model_and_diffusion_general_skeleton(args):
    model = GMDM(**get_gmdm_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_model_and_diffusion_baseline(args, cond_dict):
    model = MDMBaseline(**get_baseline_args(args, cond_dict))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_model_args(args, data):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    max_joints=0 #irrelevant
    feature_len=0 #irrelevant
    per_head_spatial_pe = False
    dataset = args.dataset
    if hasattr(args, 'object_type') and args.object_type != 'Human':
        dataset = 'truebones'

    if dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1

    elif dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    elif dataset=='truebones':
        cond_mode = 'object_type'
        max_joints=args.max_joints
        feature_len=25
        data_rep = 'truebones_tensor'
        njoints = 263 # irrelevant
        nfeats = 1 #irrelevant
        
    elif dataset=='humanml_mat':
        cond_mode = 'object_type'
        max_joints=22
        feature_len=25
        data_rep = 'truebones_tensor'
        njoints = 263 # irrelevant
        nfeats = 1 #irrelevant


    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'max_joints': max_joints, 
            'feature_len':feature_len, 'per_head_spatial_pe': per_head_spatial_pe, 'masked_attn_from_step': args.masked_attn_from_step}

def get_baseline_args(args, cond_dict):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6
    max_joints=0
    feature_len=0
    per_head_spatial_pe = False
    dataset = args.dataset
    if hasattr(args, 'object_type') and args.object_type != 'Human':
        dataset = 'truebones'

    if dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1

    elif dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    elif dataset=='truebones':
        max_joints = njoints = 143
        # feature_len=25
        data_rep = 'truebones_tensor'
        nfeats = 13
        
    elif dataset=='humanml_mat':
        max_joints=22
        feature_len=25
        data_rep = 'truebones_tensor'
        njoints = 263
        nfeats = 1


    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': None,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'max_joints': max_joints, 
            'feature_len':feature_len, 'per_head_spatial_pe': per_head_spatial_pe, 'masked_attn_from_step': args.masked_attn_from_step,
            'concat_tpos_first_frame': args.concat_tpos_first_frame, 'linear_layer_per_object_type': args.linear_layer_per_object_type,
            'cond_dict': cond_dict}

def get_gmdm_args(args):
    t5_model_dim = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    cond_mode = get_cond_mode(args)
    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    feature_len=13 #irrelevant
    cond_mode = 'object_type'
    feature_len=13

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec,  'dataset': args.dataset, 'max_joints': max_joints, 
            'feature_len':feature_len, 'masked_attn_from_step': args.masked_attn_from_step, 'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 4}

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 100
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
        lambda_ric_pos=args.lambda_ric_pos
    )