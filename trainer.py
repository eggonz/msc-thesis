import os
import shutil
import argparse
import datetime
import random
from pprint import pprint
from colorama import Style

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from src.expert_mixers import get_learned_mixer_class
from src.expert_dataset import ExpertDataset
from src.utils.loggers import TrainerLogger
from src.utils.pose_ops import pose_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.', default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--wandb", action="store_true", help="force wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="path to output folder to resume training. if provided, config is ignored.")
    args = parser.parse_args()
    assert args.resume is None or os.path.exists(args.resume), 'args.resume does not exist'
    assert args.config is None or os.path.exists(args.config), 'args.config does not exist'
    assert args.config is not None or args.resume is not None, 'either args.config or args.resume must be provided'
    return args


def load_config(config_path, args):
    cfg = config.load_config(config_path)
    if args.debug:
        cfg['expname'] = 'DEBUG_' + cfg['expname']
    assert cfg['train']['mix_method'] in [
        'learned_slerp',
        'learned_linear',
        'learned_matrix',
        'learned_lie',
        'learned_pose',
        ], cfg['train']['mix_method']
    for mode_key in ('train', 'test'):
        pose_hist = cfg.get(mode_key, {}).get('data_gen', {}).get('pose_init', {}).get('pose_hist')
        mapper_pose = cfg.get(mode_key, {}).get('data_gen', {}).get('pose_mixing', {}).get('mapped_pose')
        if pose_hist:
            assert pose_hist in ['mix', 'gt'], f'{mode_key}:pose_init:pose_hist={pose_hist}'
        if mapper_pose:
            assert mapper_pose in ['mix', 'gt'], f'{mode_key}:pose_mixing:mapped_pose={mapper_pose}'
    return cfg


class Trainer:
    def __init__(self):
    
        time_start = datetime.datetime.now()

        print(Style.RESET_ALL)
        torch.multiprocessing.set_start_method('spawn')

        self.args = parse_args()

        if self.args.resume:
            print(f"Resuming training from {self.args.resume}")
            self.cfg = load_config(f'{self.args.resume}/cfg.yaml', self.args)
            self.output = self.args.resume
        else:
            self.cfg = load_config(self.args.config, self.args)
            time_string = time_start.isoformat()
            self.output = os.path.join(self.cfg["output"], self.cfg["setting"], self.cfg["expname"], time_string)
            os.makedirs(self.output, exist_ok=True)
            config.save_config(self.cfg, f'{self.output}/cfg.yaml')
        
        setup_seed(self.cfg['setup_seed'])

        if self.args.debug:
            print("**DEBUG MODE ON**")
            print(f'{self.args=}')
            pprint(self.cfg)

        self.logger = TrainerLogger()

        self.device = self.args.device

        self.scenes = self.get_scenes_list()

        self.reps = self.cfg['train']['inner_reps']
        self.folds = self.cfg['train']['folds']
        self.epochs = self.cfg['train']['epochs']
        self.batch_size = self.cfg['train']['batch_size']
        self.val_batch_size = self.cfg['train']['val_batch_size']
        self.lr = self.cfg['train']['lr']
        self.mix_method = self.cfg['train']['mix_method']

        os.makedirs(f'{self.output}/slam_runs', exist_ok=True)
        os.makedirs(f'{self.output}/saved_data', exist_ok=True)
        os.makedirs(f'{self.output}/ckpts', exist_ok=True)

        self.test_config = self.cfg.get('test')

        self.latest_ckpt_path = f'{self.output}/ckpts/mixer_latest.pth'
        self.status_file = f'{self.output}/status.txt'
        self.resume_start = {'scene': 0, 'rep': 0}
        if self.args.resume:
            try:
                with open(self.status_file, 'r') as f:
                    contents = f.read().strip().split()
            except FileNotFoundError as e:
                print('No status file found. Cannot resume.')
                raise e
            if len(contents) == 2:  # 'init', 'finished' or 'scene rep'
                s, r = contents
                next_r = (int(r) + 1) % self.reps
                next_s = int(s) + (int(r) + 1) // self.reps  # can be more than len(scenes)
                self.resume_start = {'scene': next_s, 'rep': next_r}

    def loss_fn(self, mix_delta, gt_delta):
        #return torch.nn.functional.mse_loss(mix_delta, gt_delta)
        return pose_loss(mix_delta, gt_delta)

    def get_scenes_list(self):
        scenes_path = f'{self.output}/scenes.txt'
        if self.args.resume:
            with open(scenes_path, 'r') as f:
                scenes = f.read().strip().splitlines()
            print('Loaded scenes:', scenes)
        else:
            # get scenes randomized
            scenes = self.cfg['train']['scenes']
            outer_reps = self.cfg['train']['outer_reps']
            scenes = [s for s in scenes for _ in range(outer_reps)]
            random.shuffle(scenes)
            # save scenes
            with open(scenes_path, 'w') as f:
                for scene in scenes:
                    f.write(scene + '\n')
        return scenes

    def get_scene_args(self, scene_cfg_path):
        """copies structure from run.py>parse_args()"""
        return argparse.Namespace(
            config=scene_cfg_path,
            device=self.args.device,
            max_frames=None,
            only_tracking=False,
            make_video=True,  # for evaluation to work
            image_size=None,
            calibration_txt=None,
            mode=None,
            debug=self.args.debug,
            expname=None,
            wandb=self.args.wandb,
        )
    
    def load_scene_config(self, scene_cfg_path, scene_idx=None, is_test=False):
        cfg = config.load_config(scene_cfg_path, './configs/expert_slam.yaml')

        # overwrite config with trainer config
        cfg['data']['output'] = self.cfg['output']  # ignored by slam
        cfg['setting'] = self.cfg['setting']
        cfg['expname'] = self.cfg['expname']
        if is_test:
            cfg['expname'] += '-test'
        if scene_idx is not None:
            cfg['expname'] += f'-S{scene_idx:02d}-{cfg["dataset"]}-{cfg["scene"]}'
        cfg['tracking']['pose_mixing']['method'] = self.mix_method

        # overwrite slam config, if present in trainer_cfg
        mode_key = 'test' if is_test else 'train'
        pose_hist = self.cfg.get(mode_key, {}).get('data_gen', {}).get('pose_init', {}).get('pose_hist')
        add_noise_scale = self.cfg.get(mode_key, {}).get('data_gen', {}).get('pose_init', {}).get('add_noise_scale')
        mapped_pose = self.cfg.get(mode_key, {}).get('data_gen', {}).get('pose_mixing', {}).get('mapped_pose')
        if pose_hist:
            cfg['tracking']['pose_init']['pose_hist'] = pose_hist
        if add_noise_scale:
            cfg['tracking']['pose_init']['add_noise_scale'] = add_noise_scale
        if mapped_pose:
            cfg['tracking']['pose_mixing']['mapped_pose'] = mapped_pose

        # fill in missing values
        if self.args.wandb:
            cfg['wandb'] = True
        
        # check input
        assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
        assert cfg['tracking']['pose_init']['pose_hist'] in ['f2f', 'f2m', 'mix', 'gt'], cfg['tracking']['pose_init']['pose_hist']
        assert cfg['tracking']['f2f']['pose_filling'] in ['slerp', 'slerp_ba'], cfg['tracking']['f2f']['pose_filling']
        # assert cfg['tracking']['pose_mixing']['method']
        assert cfg['tracking']['pose_mixing']['psnr_criterion_metric'] in ['psnr', 'psnr_masked'], cfg['tracking']['pose_mixing']['psnr_criterion_metric']
        assert cfg['tracking']['pose_mixing']['mapped_pose'] in ['mix', 'gt'], cfg['tracking']['pose_mixing']['mapped_pose']
        return cfg
    
    def gen_slam_data(self, scene_idx, scene_cfg_path, mixer, is_test=False):
        args = self.get_scene_args(scene_cfg_path)
        cfg = self.load_scene_config(args.config, scene_idx=scene_idx, is_test=is_test)

        save_pkl_path = f'{self.output}/saved_data/S{scene_idx:02d}.pkl'

        stream = get_dataset(cfg, device=self.device)

        DEBUG_RESUME_DATA = None
        # DEBUG_RESUME_DATA = "/gpfs/work5/0/prjs0799/expert_output/ExpertTrainer/learned_expert_pae/pae_replica/2024-11-27T00:10:00.668379/saved_data/S00.pkl"

        if self.args.resume and scene_idx < self.resume_start['scene']:
            raise ValueError(f'Trying to resume from an old scene. {scene_idx=} < {self.resume_start["scene"]=}')

        if self.args.resume and scene_idx == self.resume_start['scene'] and os.path.exists(save_pkl_path):
            print('===', 'Resume: Loading Expert Data', '===')
            expert_dataset = ExpertDataset.load_pkl(save_pkl_path)

        elif DEBUG_RESUME_DATA:  # DEBUG
            print('===', 'DEBUG: Loading Expert Data', '===')
            expert_dataset = ExpertDataset.load_pkl(DEBUG_RESUME_DATA)

        else:
            print('===', 'Running SLAM', '===')
    
            time_start = datetime.datetime.now()

            if args.debug:
                print(f'{args=}')
                pprint(cfg)

            output_dir = f'{self.output}/slam_runs/S{scene_idx:02d}-{cfg["dataset"]}-{cfg["scene"]}'
            if os.path.exists(output_dir):
                # remove old data if exists
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f'Output directory: {output_dir}')

            # run SLAM and generate data
            mixer.eval()
            slam = SLAM(args, cfg, output_dir=output_dir)
            slam.run(stream, mixer)

            time_end = datetime.datetime.now()
            print(f'Finished in {time_end - time_start}')

            expert_dataset = ExpertDataset(slam.expert_data_saver.get_data())
            expert_dataset.save_pkl(save_pkl_path)

        print('ExpertDataset length:', len(expert_dataset))

        return expert_dataset, stream
    
    def save_state(self, mixer=None, scene=None, rep=None, init=False, final=False):
        mixer.save(self.latest_ckpt_path)
        if init:
            mixer.save(f'{self.output}/ckpts/mixer_init.pth')
            with open(self.status_file, 'w') as f:
                f.write(f'init')
        elif final:
            mixer.save(f'{self.output}/ckpts/mixer_final.pth')
            with open(self.status_file, 'w') as f:
                f.write(f'finished')
        else:
            mixer.save(f'{self.output}/ckpts/mixer_S{scene:02d}R{rep:02d}.pth')
            with open(self.status_file, 'w') as f:
                f.write(f'{scene} {rep}')

    def run(self):

        MixerClass = get_learned_mixer_class(self.mix_method)
        mixer = MixerClass(self.device)
        if self.args.resume:
            mixer.load(self.latest_ckpt_path)
        else:
            self.save_state(mixer, init=True)

        for s, scene_cfg_path in enumerate(self.scenes):
            if s < self.resume_start['scene']:
                print('===', f'Skipping Scene:{s+1}/{len(self.scenes)}', '===')
                continue

            print('===', f'Scene:{s+1}/{len(self.scenes)}', f'({scene_cfg_path})', '===')

            expert_dataset, stream = self.gen_slam_data(s, scene_cfg_path, mixer)

            print('===', 'Training Mixer', '===')

            t1 = datetime.datetime.now()

            # Iterate Reps
            for rep in range(self.reps):
                if rep < self.resume_start['rep']:
                    print('===', f'Skipping Rep:{rep+1}/{self.reps}', '===')
                    continue

                kf = KFold(n_splits=self.folds, shuffle=True)#, random_state=42)

                best_fold_loss = float('inf')  # TODO choose best metric for mixer.net performance
                best_fold_metrics = None
                best_mixer = None

                # Iterate Folds
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(expert_dataset)):
                    self.logger.update(scene=s, rep=rep, fold=fold_idx)

                    print('===', f'Rep:{rep+1}/{self.reps}', f'Fold:{fold_idx+1}/{self.folds}', '===')
                    
                    # Define the data loaders for the current fold
                    train_loader = DataLoader(
                        dataset=expert_dataset,
                        batch_size=self.batch_size,
                        sampler=SubsetRandomSampler(train_idx),
                    )

                    val_loader = DataLoader(
                        dataset=expert_dataset,
                        batch_size=self.val_batch_size,
                        sampler=SubsetRandomSampler(val_idx),
                    )

                    trained_mixer, fold_loss, fold_metrics = self.train_fold(train_loader, val_loader, mixer.clone(), stream)

                    if fold_loss < best_fold_loss:
                        best_fold_loss = fold_loss
                        best_fold_metrics = fold_metrics
                        best_mixer = trained_mixer.clone()

                    torch.cuda.empty_cache()

                    self.logger.log(  # log at fold level
                        scene=s,
                        rep=rep,
                        fold=fold_idx,
                        epoch=None,
                        batch=None,
                        loss_fold=fold_loss,
                        **{f'val_{m}_fold': v for m, v in fold_metrics.items()},
                    )

                # save ckpt and status
                mixer = best_mixer#.clone()
                self.save_state(mixer, scene=s, rep=rep)

                print('===', f'Finished S{s:02d}R{rep:02d}.', 'Performance:', best_fold_loss, '===')

                self.logger.log(  # log at rep level
                    scene=s,
                    rep=rep,
                    fold=None,
                    epoch=None,
                    batch=None,
                    loss=best_fold_loss,
                    **{f'val_{m}': v for m, v in best_fold_metrics.items()},
                )

                # save training progress data
                self.logger.save_csv(f'{self.output}/trainer_log.csv')

                torch.cuda.empty_cache()

            t2 = datetime.datetime.now()
            print(f"Scene training time: {t2-t1}")
        
        # save final state
        self.save_state(mixer, final=True)

        if self.test_config:
            print('===', 'Testing', '===')

            self.output = os.path.join(self.output, 'test')
            os.makedirs(self.output, exist_ok=True)

            test_scenes = self.test_config['scenes']
            test_batch_size = self.test_config['batch_size']

            os.makedirs(f'{self.output}/slam_runs', exist_ok=True)
            os.makedirs(f'{self.output}/saved_data', exist_ok=True)

            # Test
            scene_losses = {}
            
            for s, scene_cfg_path in enumerate(test_scenes):
                
                print('===', f'Scene:{s+1}/{len(test_scenes)}', f'({scene_cfg_path})', '===')

                expert_dataset, stream = self.gen_slam_data(s, scene_cfg_path, mixer, is_test=True)

                print('===', 'Evaluating mixer', '===')

                test_loader = DataLoader(
                    dataset=expert_dataset,
                    batch_size=test_batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=False
                )

                test_loss, test_metrics = self.test(test_loader, mixer, stream)

                scene_losses[s] = test_loss
                
                torch.cuda.empty_cache()

            print('===', 'Evaluation finished', '===')
            tot_loss = sum(scene_losses.values())
            avg_loss = tot_loss / len(test_scenes)
            print('Total   Test Loss:', tot_loss)
            print('Average Test Loss:', avg_loss)

            # save test results
            with open(f'{self.output}/results.txt', 'w') as f:
                f.write(f'checkpoint: {self.latest_ckpt_path}\n')
                for s in range(len(test_scenes)):
                    f.write(f's{s:02d}_test_loss: {scene_losses[s]}\n')
                f.write(f'tot_test_loss: {tot_loss}\n')
                f.write(f'avg_test_loss: {avg_loss}\n')
                f.write('####################\n')

        print("Done! :)")
    
    def train_fold(self, train_loader, val_loader, mixer, stream):
        """mixer: mixer to initialize training"""

        mixer = mixer.clone()
        optimizer = torch.optim.AdamW(mixer.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        best_val_metrics = None
        best_mixer = None

        # Iterate Epochs
        for epoch in range(self.epochs):

            # Training
            train_loss = self.train_epoch(train_loader, mixer, optimizer, stream, epoch)

            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader, mixer, stream)

            self.logger.log(  # log at epoch level
                epoch=epoch,
                batch=None,
                loss_epoch=train_loss,
                val_loss_epoch=val_loss,
            )

            # Keep best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics
                best_mixer = mixer.clone()

        torch.cuda.empty_cache()
        print("cuda memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB (max", torch.cuda.max_memory_allocated() / 1024**3, "GB). cached: ", torch.cuda.memory_reserved() / 1024**3, "GB (max", torch.cuda.max_memory_reserved() / 1024**3, "GB)")

        return best_mixer, best_val_loss, best_val_metrics

    def train_epoch(self, train_loader, mixer, optimizer, stream, epoch):
        
        mixer.train()

        ema_loss = None
        loss = float('inf')

        # Iterate Batches
        pb = tqdm(train_loader) if self.args.verbose else train_loader
        for batch_idx, batch in enumerate(pb):
            if self.args.verbose:
                pb.set_description(f"Epoch {epoch+1}/{self.epochs} loss: {loss:.4f}")

            # Train batch
            timestamp, image, depth, init_pose, f2f_delta, f2m_delta, gt_delta, prev_image, prev_depth = self.get_batch_data(batch, stream)
            mix_delta = mixer(timestamp, image, depth, init_pose, f2f_delta, f2m_delta, prev_image, prev_depth)
            loss = self.loss_fn(mix_delta, gt_delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            # TODO train metrics

            if ema_loss is None:
                ema_loss = loss
            else:
                gamma = 2 / (batch_idx + 2)  # window is num of batches (batch_idx + 1)
                ema_loss = gamma * loss + (1 - gamma) * ema_loss

            if batch_idx == len(train_loader) - 1:
                epoch_train_loss = ema_loss
                if self.args.verbose:
                    pb.set_description(f"Epoch {epoch+1}/{self.epochs} loss: {epoch_train_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs} loss: {epoch_train_loss:.4f}")

            self.logger.log(  # log at batch level
                epoch=epoch,
                batch=batch_idx,
                loss_batch=loss,
                loss_batch_ema=ema_loss,
            )

            torch.cuda.empty_cache()

        epoch_train_loss = ema_loss
        return epoch_train_loss


    @torch.no_grad()
    def validate_epoch(self, val_loader, mixer, stream):
        """mixer: trained_mixer"""

        mixer.eval()

        cum_loss = 0
        loss = 0

        # Iterate Batches
        pb = tqdm(val_loader) if self.args.verbose else val_loader
        for batch_idx, batch in enumerate(pb):
            if self.args.verbose:
                pb.set_description(f"Validation loss: {loss:.4f}")
            
            # Validate batch
            timestamp, image, depth, init_pose, f2f_delta, f2m_delta, gt_delta, prev_image, prev_depth = self.get_batch_data(batch, stream)
            mix_delta = mixer(timestamp, image, depth, init_pose, f2f_delta, f2m_delta, prev_image, prev_depth)
            loss = self.loss_fn(mix_delta, gt_delta)
            loss = loss.item()

            # TODO val metrics

            cum_loss += loss

            if batch_idx == len(val_loader) - 1:
                val_loss = cum_loss / len(val_loader)
                if self.args.verbose:
                    pb.set_description(f"Validation loss: {val_loss:.4f}")
                else:
                    print(f"Validation loss: {val_loss:.4f}")

            torch.cuda.empty_cache()

        val_loss = cum_loss / len(val_loader)
        val_metrics = {}
        return val_loss, val_metrics

    @torch.no_grad()
    def test(self, test_loader, mixer, stream):
        t1 = datetime.datetime.now()
        mixer.eval()
        cum_loss = 0
        pb = tqdm(test_loader, desc="Testing") if self.args.verbose else test_loader
        for batch in pb:
            timestamp, image, depth, init_pose, f2f_delta, f2m_delta, gt_delta, prev_image, prev_depth = self.get_batch_data(batch, stream)
            mix_delta = mixer(timestamp, image, depth, init_pose, f2f_delta, f2m_delta, prev_image, prev_depth)
            loss = self.loss_fn(mix_delta, gt_delta)
            cum_loss += loss.item()
            # TODO test metrics
            torch.cuda.empty_cache()
        test_loss = cum_loss / len(test_loader)
        t2 = datetime.datetime.now()
        print(f"Test time: {t2-t1}")
        print(f"Test loss: {test_loss:.4f}")
        test_metrics = {}
        return test_loss, test_metrics

    def get_batch_data(self, batch, stream):
        """retireve data from batch in correct format

        Returns:
            timestamp: torch.Tensor [B,]
            image: torch.Tensor [B,3,H,W]
            depth: torch.Tensor [B,1,H,W]
            init_pose: torch.Tensor [B,7]
            f2f_delta: torch.Tensor [B,7]
            f2m_delta: torch.Tensor [B,7]
            gt_delta: torch.Tensor [B,7]
            prev_image: torch.Tensor [B,3,H,W]
            prev_depth: torch.Tensor [B,1,H,W]
        """

        idx, init_pose, f2f_delta, f2m_delta, gt_delta = batch  # [B,], 4x[B,7]
        init_pose = init_pose.to(self.device)
        f2f_delta = f2f_delta.to(self.device)
        f2m_delta = f2m_delta.to(self.device)
        gt_delta = gt_delta.to(self.device)

        timestamps = []
        images = []
        depths = []
        prev_images = []
        prev_depths = []
        for i in idx:
            # NOTE: change with expert-resolution
            # timestamp, image, gt_depth, intrinsics, gt_c2w, gt_color_full, gt_depth_full = stream[i]
            timestamp, image, gt_depth, _, _, _, _ = stream[i]
            _, prev_image, prev_gt_depth, _, _, _, _ = stream[i-1]
            gt_color = image.squeeze(0).permute(1,2,0)  # [1,3,h,w] -> [h,w,3]
            prev_gt_color = prev_image.squeeze(0).permute(1,2,0)  # [1,3,h,w] -> [h,w,3]
            timestamps.append(timestamp)  # [1,]
            images.append(gt_color)  # [H,W,3]
            depths.append(gt_depth)  # [H,W]
            prev_images.append(prev_gt_color)  # [H,W,3]
            prev_depths.append(prev_gt_depth)  # [H,W]

        timestamps = torch.tensor(timestamps, device=self.device)  # [B,]
        images = torch.stack(images).permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]
        depths = torch.stack(depths).unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        prev_images = torch.stack(prev_images).permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]
        prev_depths = torch.stack(prev_depths).unsqueeze(1)  # [B,H,W] -> [B,1,H,W]

        return timestamps, images, depths, init_pose, f2f_delta, f2m_delta, gt_delta, prev_images, prev_depths


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
