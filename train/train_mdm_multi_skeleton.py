# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on motions.
"""
import sys
import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion_general_skeleton
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation

def main():
    DEBUG = sys.gettrace() is not None
    args = train_args()
    fixseed(args.seed)
    save_dir = args.save_dir
    if save_dir is None:
        prefix = "arch"
        if args.model_prefix is not None:
            prefix = args.model_prefix
        model_name = f'{prefix}_{args.arch.replace("_",".")}_dataset_{args.dataset.replace("_",".")}_bs_{args.batch_size}_latentdim_{args.latent_dim}'
        mod_list = [m for m in os.listdir(os.path.join(os.getcwd(), 'save')) if m.startswith(model_name)]
        if len(mod_list) > 0 and not args.overwrite:
            model_name = f'{model_name}_{len(mod_list)}'
        save_dir = os.path.join(os.getcwd(), 'save' ,model_name)
        args.save_dir = save_dir
        
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(save_dir, resume=bool(args.resume_checkpoint), debug=DEBUG)
    train_platform.report_args(args, name='Args')

    if save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(save_dir))
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args_path = os.path.join(save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, debug=DEBUG, temporal_window=args.temporal_window, hml=args.hml, t5_name='t5-base', balanced=args.balanced, tpos_not_normalized=args.tpos_not_normalized, objects_subset=args.objects_subset)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())

    train_platform.watch_model(model)
 
    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
