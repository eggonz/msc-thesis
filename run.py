import argparse
import numpy as np
import os
import random
import shutil
import torch
import datetime
from colorama import Style
from pprint import pprint

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_source_code(backup_directory):
    return 
    ignore_hidden = shutil.ignore_patterns(
        '.', '..', '.git*', '*pycache*', '*build', '*.fuse*', '*_drive_*',
        '*pretrained*', '*output*', '*media*', '*.so', '*.pyc', '*.Python',
        '*.eggs*', '*.DS_Store*', '*.idea*', '*.pth', '*__pycache__*', '*.ply',
        '*exps*',
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system('chmod -R g+w {}'.format(backup_directory))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_frames", type=int, default=None, help="Only [0, max_frames] Frames will be run")
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--make_video", action="store_true", help="to generate video as in our project page")
    parser.add_argument("--image_size", nargs='+', default=None,
                        help='image height and width, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--calibration_txt', type=str, default=None,
                        help='calibration parameters: fx, fy, cx, cy, this have higher priority, '
                             'can overwrite the one in config file')
    parser.add_argument('--mode', type=str,
                        help='slam mode: mono, rgbd or stereo')
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--expname", type=str, default='', help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="force wandb logging")
    args = parser.parse_args()
    return args


def load_config(config_path, args):
    cfg = config.load_config(config_path, './configs/expert_slam.yaml')
    setup_seed(cfg['setup_seed'])
    if args.max_frames is not None:
        cfg['max_frames'] = args.max_frames
    if args.expname:
        cfg['expname'] = args.expname
    if args.debug:
        cfg['expname'] = 'DEBUG_' + cfg['expname']
    if args.wandb:
        cfg['wandb'] = True
    if args.mode is not None:
        cfg['mode'] = args.mode
    if args.only_tracking:
        cfg['only_tracking'] = True
        cfg['wandb'] = False
        cfg['mono_prior']['predict_online'] = True
    if args.image_size is not None:
        cfg['cam']['H'], cfg['cam']['W'] = args.image_size
    if args.calibration_txt is not None:
        cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] = np.loadtxt(args.calibration_txt).tolist()
    assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
    assert cfg['tracking']['pose_init']['pose_hist'] in ['f2f', 'f2m', 'mix', 'gt'], cfg['tracking']['pose_init']['pose_hist']
    assert cfg['tracking']['f2f']['pose_filling'] in ['slerp', 'slerp_ba'], cfg['tracking']['f2f']['pose_filling']
    assert cfg['tracking']['pose_mixing']['method'] in [
        'f2f', 'f2m', 'mean', 'best', 'gt', 'psnr',
        'learned_slerp',
        'learned_linear',
        'learned_matrix',
        'learned_lie',
        'learned_pose',
        ], cfg['tracking']['pose_mixing']['method']
    assert cfg['tracking']['pose_mixing']['psnr_criterion_metric'] in ['psnr', 'psnr_masked'], cfg['tracking']['pose_mixing']['psnr_criterion_metric']
    assert cfg['tracking']['pose_mixing']['mapped_pose'] in ['mix', 'gt'], cfg['tracking']['pose_mixing']['mapped_pose']
    return cfg


if __name__ == '__main__':
    
    time_start = datetime.datetime.now()

    print(Style.RESET_ALL)
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()
    print(f'{args=}')
    cfg = load_config(args.config, args)
    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")

    if args.debug:
        print("**DEBUG MODE ON**")

    # use separate output directories for each experiment
    time_string = time_start.isoformat()
    output_dir = os.path.join(cfg["data"]["output"], cfg["expname"], cfg['setting'], cfg['scene'], time_string)

    print(f'Output directory: {output_dir}')

    os.makedirs(output_dir, exist_ok=True)
    backup_source_code(os.path.join(output_dir, 'code'))

    if args.debug:
        pprint(cfg)

    dataset_stream = get_dataset(cfg, device=args.device)

    slam = SLAM(args, cfg, output_dir=output_dir)
    slam.run(dataset_stream, None)

    print('Done!')
    time_end = datetime.datetime.now()
    print(f'Total time: {time_end - time_start}')
