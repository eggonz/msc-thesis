import os
import argparse
import datetime
from pprint import pprint
from colorama import Style

import torch
from torch.utils.data import DataLoader

from src import config
from src.expert_mixers import load_learned_mixer_ckpt
from trainer import Trainer, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--wandb", action="store_true", help="force wandb logging")
    args = parser.parse_args()
    return args


def load_config(config_path, args):
    cfg = config.load_config(config_path)
    if args.debug:
        cfg['expname'] = 'DEBUG_' + cfg['expname']
    if args.wandb:
        cfg['wandb'] = True
    if args.ckpt is not None:  # override ckpt
        cfg['test']['ckpts'] = [args.ckpt]
    pose_hist = cfg.get('test', {}).get('data_gen', {}).get('pose_init', {}).get('pose_hist')
    mapper_pose = cfg.get('test', {}).get('data_gen', {}).get('pose_mixing', {}).get('mapped_pose')
    if pose_hist:
        assert pose_hist in ['mix', 'gt'], pose_hist
    if mapper_pose:
        assert mapper_pose in ['mix', 'gt'], mapper_pose
    return cfg


class Tester(Trainer):
    def __init__(self):
    
        time_start = datetime.datetime.now()

        print(Style.RESET_ALL)
        torch.multiprocessing.set_start_method('spawn')

        self.args = parse_args()
        self.cfg = load_config(self.args.config, self.args)
        setup_seed(self.cfg['setup_seed'])

        if self.args.debug:
            print("**DEBUG MODE ON**")
            print(f'{self.args=}')
            pprint(self.cfg)

        self.args.resume = False  # integration with trainer
        
        time_string = time_start.isoformat()
        self.output = os.path.join(self.cfg["output"], self.cfg["setting"], self.cfg["expname"], time_string)
        os.makedirs(self.output, exist_ok=True)

        config.save_config(self.cfg, f'{self.output}/cfg.yaml')

        self.device = self.args.device

        self.mix_method = None  # specific to each mixer ckpt loaded

        self.scenes = self.cfg['test']['scenes']
        self.ckpts = self.cfg['test'].get('ckpts', [])
        self.batch_size = self.cfg['test']['batch_size']

        if not self.ckpts:
            raise ValueError('No checkpoint specified for testing!')
        
        os.makedirs(f'{self.output}/slam_runs', exist_ok=True)
        os.makedirs(f'{self.output}/saved_data', exist_ok=True)

    def run(self):

        for n, ckpt_path in enumerate(self.ckpts):

            if len(self.ckpts) > 1:
                print('===', 'Testing for Checkpoint', f'{n+1}/{len(self.ckpts)}', '===')

            # Load mixer
            mixer = load_learned_mixer_ckpt(ckpt_path, self.device)
            mixer.eval()
            self.mix_method = mixer.mix_method
            print('Mixer loaded from', ckpt_path)
            print('Mixer type:', self.mix_method)

            scene_losses = {}
            
            for s, scene_cfg_path in enumerate(self.scenes):
                
                print('===', f'Scene:{s+1}/{len(self.scenes)}', f'({scene_cfg_path})', '===')

                expert_dataset, stream = self.gen_slam_data(s, scene_cfg_path, mixer, is_test=True)

                print('===', 'Evaluating mixer', '===')

                test_loader = DataLoader(
                    dataset=expert_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=False,
                )

                test_loss, test_metrics = self.test(test_loader, mixer, stream)

                scene_losses[s] = test_loss

            print('===', 'Evaluation finished', '===')
            tot_loss = sum(scene_losses.values())
            avg_loss = tot_loss / len(self.scenes)
            print('Total Test Loss:', tot_loss)
            print('Average Test Loss:', avg_loss)

            # save test results
            with open(f'{self.output}/test_results.txt', 'a') as f:
                f.write(f'checkpoint: {ckpt_path}\n')
                for s in range(len(self.scenes)):
                    f.write(f's{s:02d}_test_loss: {scene_losses[s]}\n')
                f.write(f'tot_test_loss: {tot_loss}\n')
                f.write(f'avg_test_loss: {avg_loss}\n')
                f.write('####################\n')

        print('Done! :)')


if __name__ == '__main__':
    tester = Tester()
    tester.run()