import numpy as np
import os
import logging
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from tqdm import tqdm
import lietorch

from src.common import get_tensor_from_camera
from src.droid_backend import DroidBackend as Backend
from src.droid_frontend import DroidFrontend as Frontend
from src.motion_filter import MotionFilter
from src.utils.loggers import logger
from src.utils import pose_ops
from src.utils.eval_poses import PoseErrorEvaler
from src.utils.pose_ops import pose_loss
from src.f2m_tracker import F2MTracker
from src.utils.eval_psnr import PsnrEvaler
from src.expert_mixers import get_fixed_mixer
from src.utils.pose_trajectory import PsnrExpertHist


class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.args = slam.args
        self.cfg = slam.cfg
        self.device = slam.device
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.only_tracking = slam.only_tracking
        self.offline_mapping = slam.offline_mapping
        self.output = slam.output

        self.wandb_logger = slam.wandb_logger
        if self.args.debug:
            logger.setLevel(logging.DEBUG)

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']

        # F2F
        self.frontend = Frontend(self.net, self.video, self.args, self.cfg)
        self.online_ba = Backend(self.net,self.video,self.args,self.cfg)

        # F2M
        self.f2m_tracker = F2MTracker(slam)
        self.pose_initializer = PoseInitializer(slam, self.device)

        # MIX
        self.mix_method = self.cfg['tracking']['pose_mixing']['method']
        self.psnr_criterion_metric = self.cfg['tracking']['pose_mixing']['psnr_criterion_metric']  # 'psnr' or 'psnr_masked'
        self.psnr_evaler = PsnrEvaler(slam, self.device, full_res=False)  # NOTE: change with psnr-resolution
        self.expert_data_saver = slam.expert_data_saver

        # Evaluation
        self.pose_evaler = PoseErrorEvaler()

        # save relative c2w tensor poses [tx,ty,tz,qx,qy,qz,qw], this is the output trajectory
        self.f2f_pose_history = slam.f2f_pose_history
        self.f2m_pose_history = slam.f2m_pose_history
        self.mix_pose_history = slam.mix_pose_history
        self.gt_pose_history = slam.gt_pose_history

        # absolute GT c2w matrix for t=0
        self.first_abs_gt_c2w = None

        self.psnrx_hist = PsnrExpertHist()

    def run_offline(self):
        graph_path = f"{self.output}/{self.cfg['offline_video']}"
        if not os.path.exists(graph_path):
            raise Exception(f"{graph_path} not exists! Must run the pipeline with 'only_tracking: True' first!")
        timestamps = np.load(graph_path)["timestamps"]  
        for i in range(self.cfg['tracking']['warmup']-1,timestamps.shape[0]):
            timestamp = int(timestamps[i])
            self.pipe.send({"is_keyframe":True, "video_idx":i,
                                "timestamp":timestamp, "end":False})
            self.pipe.recv()
        self.pipe.send({"is_keyframe":True, "video_idx":None,
                            "timestamp":None, "end":True})

    def run(self, stream, pose_mixer=None):
        if self.args.debug:
            logger.setLevel(logging.DEBUG)
        if self.offline_mapping:
            self.run_offline()
            return
        
        if self.mix_method.startswith('learned'):
            assert pose_mixer is not None, 'pose_mixer must be provided for learned mixers'
        elif self.mix_method != 'psnr':
            pose_mixer = get_fixed_mixer(self.mix_method)
        
        prev_kf_idx = -1
        curr_kf_idx = 0
        prev_ba_idx = 0

        pbar = tqdm(stream) if self.verbose else stream

        # NOTE: change with expert-resolution
        prev_gt_color = None
        prev_gt_depth = None

        for (timestamp, image, gt_depth, intrinsic, gt_c2w, gt_color_full, gt_depth_full) in pbar:  # timestamp: int, c2w: tensor([4,4]), camera_tensor: tensor([7]), video.poses: tensor([400,7])
            # timestamp: int
            # image: tensor([1,3,320,640]) float32
            # gt_depth: tensor([320,640]) float32
            # intrinsic: tensor([4,]) float32
            # gt_c2w: tensor([4,4]) float32
            # gt_color_full: tensor([680,1200,3]) float32
            # gt_depth_full: tensor([680,1200]) float32
            
            # if timestamp < 600:
            #     continue

            if self.verbose:
                print(Fore.CYAN)
                print("Processing Frame ", timestamp)
                print(Style.RESET_ALL)

            if self.wandb_logger is not None:
                self.wandb_logger.set_step(timestamp)

            gt_color = image.to(self.device).squeeze(0).permute(1,2,0)  # [1,3,h,w] -> [h,w,3]

            if timestamp==0:
                self.first_abs_gt_c2w = gt_c2w.clone()
            gt_c2w = self.first_abs_gt_c2w.inverse() @ gt_c2w  # relative to first frame camera pose
            # from here on, gt_c2w is relative to first frame camera pose

            gt_pose = get_tensor_from_camera(gt_c2w, Tquad=True)  # [T,quad]

            ################################################################################################

            # Motion Filter: check if the frame is a keyframe, add to video if it is. might increase video.counter
            with torch.no_grad():
                self.motion_filter.track(timestamp, image, gt_depth, intrinsic)

            # KF found by motion_filter?
            curr_kf_idx = self.video.counter.value - 1
            found_new_kf = (self.video.counter.value - 1 != prev_kf_idx)
            if found_new_kf:
                logger.debug(f'New KF candidate at {timestamp=} {curr_kf_idx=}')

            logger.debug(f'gt_pose={gt_pose.cpu().numpy().round(4)}')

            # INIT POSE
            init_pose = self.pose_initializer.get_pose(timestamp)

            if self.verbose:
                print(Fore.CYAN)
                print("Expert 1: F2F" if self.mix_method == 'psnr' else "F2F")
                print(Style.RESET_ALL)

            # >>> Expert1: F2F

            # set initial pose for F2F
            self.video.set_pose_tensor(self.video.counter.value - 1, init_pose)

            # Droid Frontend: track the frame, might decrease video.counter
            with torch.no_grad():
                f2f_pose = self.frontend(timestamp, image, intrinsic)

            logger.debug(f'f2f_pose={f2f_pose.cpu().numpy().round(4) if f2f_pose is not None else None}')

            # <<< F2F

            # Still KF after frontend?
            curr_kf_idx = self.video.counter.value - 1
            if found_new_kf and curr_kf_idx == prev_kf_idx:
                logger.debug("F2F removed keyframe!")
                found_new_kf = False
            elif found_new_kf:
                logger.info(f'KF at {timestamp=} {curr_kf_idx=}')
            else:
                logger.debug(f'non-KF at {timestamp=} {curr_kf_idx=}')

            if self.verbose:
                print(Fore.CYAN)
                print("Expert 4: F2M (RGBD)" if self.mix_method == 'psnr' else "F2M")
                print(Style.RESET_ALL)

            # >>> Expert4: F2M (RGBD)

            if f2f_pose is not None and self.cfg['tracking']['f2m']['init_from_f2f']:
                logger.debug(f'init F2M from F2F')
                init_pose = f2f_pose.clone()

            # NOTE: change with f2m-resolution
            f2m_pose = self.f2m_tracker.track(timestamp, gt_color, gt_depth, gt_c2w, init_pose, is_kf=found_new_kf, vis=True)
            logger.debug(f'f2m_pose={f2m_pose.cpu().numpy().round(4)}')
            
            if f2m_pose is not None and torch.isnan(f2m_pose).any():
                logger.warning(f'f2m_pose is NaN at {timestamp=} {curr_kf_idx=} {f2m_pose=}')

            # <<< F2M
            
            if self.mix_method == 'psnr':
                
                # >>> Expert Prediction

                if self.verbose:
                    print(Fore.CYAN)
                    print("Expert 2: F2F + F2M (RGBD)")
                    print(Style.RESET_ALL)

                # Expert2: F2F + F2M (RGBD)
                # init from F2F output
                pose_x2 = self.f2m_tracker.track(timestamp, gt_color, gt_depth, gt_c2w, f2f_pose, is_kf=found_new_kf)  # NOTE: change with f2m-resolution
                    
                if self.verbose:
                    print(Fore.CYAN)
                    print("Expert 3: F2F + F2M (RGB)")
                    print(Style.RESET_ALL)

                # Expert3: F2F + F2M (RGB)
                # init from F2F output, RGB-only refinement
                pose_x3 = self.f2m_tracker.track(timestamp, gt_color, gt_depth, gt_c2w, f2f_pose, is_kf=found_new_kf, rgb_only=True)  # NOTE: change with f2m-resolution
                    
                if self.verbose:
                    print(Fore.CYAN)
                    print("Expert 5: F2M (RGBD) + F2M (RGB)")
                    print(Style.RESET_ALL)

                # Expert5: F2M (RGBD) + F2M (RGB)
                # init from F2M output, RGB-only refinement
                pose_x5 = self.f2m_tracker.track(timestamp, gt_color, gt_depth, gt_c2w, f2m_pose, is_kf=found_new_kf, rgb_only=True)  # NOTE: change with f2m-resolution
                    
                if self.verbose:
                    print(Fore.CYAN)
                    print("Expert prediction done")
                    print(Style.RESET_ALL)

                # <<< Expert Prediction

                # >>> Evaluate PSNR

                expert_poses = [f2f_pose, pose_x2, pose_x3, f2m_pose, pose_x5]
                expert_psnr_metrics = [dict() for _ in expert_poses]
                if timestamp > 0:
                    for i, pose in enumerate(expert_poses):
                        if pose is not None:
                            # NOTE: change with psnr-resolution
                            metrics = self.psnr_evaler(gt_color, gt_depth, pose, save_name=f'{timestamp:04d}_x{i+1}')
                        else:
                            metrics = dict()
                        expert_psnr_metrics[i] = metrics
                        logger.debug(f'PSNR-Expert{i+1}: {metrics}')

                # <<< Evaluate PSNR

            # >>> MIX

            mixer_log_info = {}

            if timestamp == 0:
                # pose fixed to Id
                mix_pose = f2m_pose.clone().detach().to(self.device)

            elif self.mix_method == 'psnr':
                # PSNR-Expert: selector based on PSNR
                psnrx_guess = np.argmax([m.get(self.psnr_criterion_metric, -1) for m in expert_psnr_metrics])
                mix_pose = expert_poses[psnrx_guess].clone().detach()
            
            else:  # ignore x3-x5 experts
                # mixers work with deltas
                f2f_delta = pose_ops.pose_sub(f2f_pose, init_pose) if f2f_pose is not None else None
                f2m_delta = pose_ops.pose_sub(f2m_pose, init_pose) if f2m_pose is not None else None
                gt_delta = pose_ops.pose_sub(gt_pose, init_pose)

                if self.mix_method.startswith('learned'):
                    # Save data for expert training and inference
                    self.expert_data_saver.append(timestamp, init_pose, f2f_delta, f2m_delta, gt_delta)
                    mix_delta, mixer_log_info = pose_mixer(timestamp,
                                                           gt_color, gt_depth,  # NOTE: change with expert-resolution
                                                           init_pose,
                                                           f2f_delta, f2m_delta,
                                                           prev_gt_color, prev_gt_depth,  # NOTE: change with expert-resolution
                                                           single_sample_raw=True, return_log_info=True)
                elif pose_mixer.is_oracle:
                    # Oracle need gt_delta
                    mix_delta = pose_mixer(f2f_delta, f2m_delta, gt_delta)
                else:
                    mix_delta = pose_mixer(f2f_delta, f2m_delta)

                mix_pose = pose_ops.pose_sum(mix_delta, init_pose) if mix_delta is not None else None

            # <<< MIX

            # >>> Evaluation

            if self.mix_method == 'psnr':
                expert_losses = [pose_loss(pose, gt_pose).item() if pose is not None else -1 for pose in expert_poses]
                f2f_loss = expert_losses[0]
                f2m_loss = expert_losses[3]
            else:
                f2f_loss = pose_loss(f2f_pose, gt_pose).item() if f2f_pose is not None else -1
                f2m_loss = pose_loss(f2m_pose, gt_pose).item() if f2m_pose is not None else -1
            mix_loss = pose_loss(mix_pose, gt_pose).item() if mix_pose is not None else -1

            self.pose_evaler.reset()

            prev_gt  = self.gt_pose_history.get_pose_at_time(timestamp-1)  # none if non existent
            prev_mix = self.mix_pose_history.get_pose_at_time(timestamp-1)  # saved final pose from previous frame, starting point for both methods

            self.pose_evaler.eval_pe(f2f_pose, gt_pose, prev_mix, prev_gt, name_prefix='f2f')
            self.pose_evaler.eval_pe(f2m_pose, gt_pose, prev_mix, prev_gt, name_prefix='f2m')
            self.pose_evaler.eval_pe(mix_pose, gt_pose, prev_mix, prev_gt, name_prefix='mix')

            if timestamp == 0:
                # no deltas
                pass

            elif self.mix_method != 'psnr':
                # deltas only available for mixers
                self.pose_evaler.eval_deltas(f2f_delta, gt_delta, name_prefix='Df2f')
                self.pose_evaler.eval_deltas(f2m_delta, gt_delta, name_prefix='Df2m')
                self.pose_evaler.eval_deltas(mix_delta, gt_delta, name_prefix='Dmix')

                self.pose_evaler.compare_deltas(f2f_delta, f2m_delta, name_prefix='Df2f-vs-Df2m')
                self.pose_evaler.compare_deltas(f2f_delta, mix_delta, name_prefix='Df2f-vs-Dmix')
                self.pose_evaler.compare_deltas(f2m_delta, mix_delta, name_prefix='Df2m-vs-Dmix')

                self.pose_evaler.dist_to_plane(gt_delta, f2f_delta, f2m_delta, name_prefix='Dgt-vs-Df2f-Df2m')

            elif self.mix_method == 'psnr':
                # selector guess metrics
                psnrx_actual = np.argmin(expert_losses)
                psnrx_success = psnrx_guess == psnrx_actual

                mix_psnr_metrics = expert_psnr_metrics[psnrx_guess]
                mix_psnr = mix_psnr_metrics.get(self.psnr_criterion_metric, -1)

            # <<< Evaluation

            # >>> Logging

            mix_pose_np = mix_pose.detach().cpu().numpy().round(4) if mix_pose is not None else None
            gt_pose_np = gt_pose.detach().cpu().numpy().round(4)

            if timestamp == 0:
                # no logging for first pose, fixed to Id
                pass

            elif self.mix_method == 'psnr':
                logger.debug(f'PSNR Expert:')
                logger.debug(f'\t{psnrx_guess=}')
                logger.debug(f'\t{psnrx_actual=}')
                logger.debug(f'\t{psnrx_success=}')
                for i, (p, m, l) in enumerate(zip(expert_poses, expert_psnr_metrics, expert_losses)):
                    p_np = p.detach().cpu().numpy().round(4) if p is not None else None
                    mval = m.get(self.psnr_criterion_metric, -1)
                    logger.debug(f'\tx{i+1}: {str(p_np):<60} (loss={l:.4f}, {self.psnr_criterion_metric}={mval:.4f})')
                logger.debug(f'\tmx: {str(mix_pose_np):<60} (loss={mix_loss:.4f}, {self.psnr_criterion_metric}={mix_psnr:.4f})')
                logger.debug(f'\tgt: {str(gt_pose_np):<60}')

            elif self.mix_method != 'psnr':
                logger.debug(f'Poses ([Txyz,Qxyzw]):')
                f2f_pose_np = f2f_pose.detach().cpu().numpy().round(4) if f2f_pose is not None else None
                f2m_pose_np = f2m_pose.detach().cpu().numpy().round(4) if f2m_pose is not None else None
                logger.debug(f'\tf2f: {str(f2f_pose_np):<60} (loss={f2f_loss:.4f})')
                logger.debug(f'\tf2m: {str(f2m_pose_np):<60} (loss={f2m_loss:.4f})')
                logger.debug(f'\tmix: {str(mix_pose_np):<60} (loss={mix_loss:.4f})')
                logger.debug(f'\tgt : {str(gt_pose_np):<60}')
                f2f_delta_np = f2f_delta.detach().cpu().numpy().round(4) if f2f_delta is not None else None
                f2m_delta_np = f2m_delta.detach().cpu().numpy().round(4) if f2m_delta is not None else None
                mix_delta_np = mix_delta.detach().cpu().numpy().round(4) if mix_delta is not None else None
                logger.debug(f'\tDf2f: {str(f2f_delta_np):<60}')
                logger.debug(f'\tDf2m: {str(f2m_delta_np):<60}')
                logger.debug(f'\tDmix: {str(mix_delta_np):<60}')

            self.pose_evaler.print_metrics(fn_logger=logger.debug)
            if self.wandb_logger:
                info = {
                    'idx_frame': int(timestamp),
                    'idx_kf': curr_kf_idx,
                    **self.pose_evaler.get_metrics(),
                    'f2f_loss': f2f_loss,
                    'f2m_loss': f2m_loss,
                    'mix_loss': mix_loss,
                    **mixer_log_info,
                }
                if self.mix_method == 'psnr' and timestamp > 0:
                    info = {
                        **info,
                        **{f'x{i+1}_{k}': v for i, m in enumerate(expert_psnr_metrics) for k, v in m.items()},
                        **{f'x{i+1}_loss': l for i, l in enumerate(expert_losses)},
                        **{f'mix_{k}': v for k, v in mix_psnr_metrics.items()},
                        'psnrx_guess': int(psnrx_guess),
                        'psnrx_actual': int(psnrx_actual),
                        'psnrx_success': int(psnrx_success),
                    }
                self.wandb_logger.log(info)

            # <<< Logging

            # >>> Send Mapping
            
            if timestamp==0 or (self.frontend.is_initialized and found_new_kf):# or (self.pose_mixer.mix_method == 'f2m' and found_new_kf):

                # set video pose to mix_pose for mapping
                if mix_pose is not None and self.cfg['tracking']['pose_mixing']['mapped_pose'] == 'mix':
                    logger.debug(f'Set video pose before mapping:')
                    logger.debug(f'\tbefore: {f2f_pose.detach().cpu().numpy().round(4) if f2f_pose is not None else None}')
                    logger.debug(f'\t after: {mix_pose.detach().cpu().numpy().round(4) if mix_pose is not None else None}')
                    self.video.set_pose_tensor(curr_kf_idx, mix_pose)
                elif self.cfg['tracking']['pose_mixing']['mapped_pose'] == 'gt':
                    logger.debug(f'Set video pose before mapping:')
                    logger.debug(f'\tbefore: {f2f_pose.detach().cpu().numpy().round(4) if f2f_pose is not None else None}')
                    logger.debug(f'\t after: {gt_pose.detach().cpu().numpy().round(4)}')
                    self.video.set_pose_tensor(curr_kf_idx, gt_pose)

                if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + 20:  # False in Expert
                    print("Online BA at keyframe",curr_kf_idx)
                    self.online_ba.dense_ba(2)
                    prev_ba_idx = curr_kf_idx
                if not self.only_tracking:
                    self.pipe.send({"is_keyframe":True, "video_idx":curr_kf_idx,
                                    "timestamp":timestamp, "end":False})
                    # MAPPING IS PROCESSING timestamp
                    self.pipe.recv()
                    self.f2m_tracker.update_para_from_mapping()

            # <<< Send Mapping

            prev_kf_idx = curr_kf_idx

            # >>> save history
            self.gt_pose_history.append(timestamp, gt_pose.detach().cpu())
            if f2f_pose is not None:
                self.f2f_pose_history.append(timestamp, f2f_pose.detach().cpu())
            if f2m_pose is not None:
                self.f2m_pose_history.append(timestamp, f2m_pose.detach().cpu())
            if mix_pose is not None:
                self.mix_pose_history.append(timestamp, mix_pose.detach().cpu())

            if self.mix_method == 'psnr' and timestamp > 0:
                self.psnrx_hist.append_gt(timestamp, gt_pose.detach().cpu())
                for e, (xpose, xmetr, xloss) in enumerate(zip(expert_poses, expert_psnr_metrics, expert_losses)):
                    xpsnr = xmetr.get(self.psnr_criterion_metric, -1)
                    self.psnrx_hist.append_expert(e, xpose, xpsnr, xloss)
                
            # NOTE: change with expert-resolution
            prev_gt_color = gt_color
            prev_gt_depth = gt_depth
            # <<< save history

            torch.cuda.empty_cache()

        if self.wandb_logger is not None:
            self.wandb_logger.set_step(timestamp+1)
        
        if not self.only_tracking:
            self.pipe.send({"is_keyframe":True, "video_idx":None,
                            "timestamp":None, "end":True})
        
        if self.mix_method == 'psnr':
            # save history
            self.psnrx_hist.save(f'{self.output}/psnrx')


class PoseInitializer:
    def __init__(self, slam, out_device):
        self.device = out_device

        cfg_f2m_pose_hist = slam.cfg['tracking']['pose_init']['pose_hist']
        self.const_speed_assumption = slam.cfg['tracking']['pose_init']['const_speed_assumption']
        self.add_noise_scale = slam.cfg['tracking']['pose_init']['add_noise_scale']

        self.pose_hist = {
            'f2f': slam.f2f_pose_history,
            'f2m': slam.f2m_pose_history,
            'mix': slam.mix_pose_history,
            'gt': slam.gt_pose_history,
        }.get(cfg_f2m_pose_hist)

    def get_pose(self, timestamp):
        """ returns a (7,) [T,quad] tensor c2w pose, before applying [:3, 1:3]*=-1 flip """
        is_first_frame = timestamp == 0 or len(self.pose_hist) == 0
        if is_first_frame:
            logger.debug(f'init c2w with identity (first timestamp)')
            pose = self.get_identity()
        elif self.const_speed_assumption and len(self.pose_hist)>=2:
            logger.debug(f'init c2w with constant speed assumption')
            pose = self.const_speed(timestamp)
        else:
            logger.debug(f'init c2w with previous pose')
            pose = self.copy_last()
        if not is_first_frame and self.add_noise_scale > 0:
            logger.debug(f'adding noise to pose')
            pose = self.add_noise(pose, self.add_noise_scale)
        logger.debug(f'pose initialized to {pose.cpu().numpy().round(4)}')
        return pose.to(self.device).float()

    def get_identity(self):
        return lietorch.SE3.Identity(1)[0].data

    def const_speed(self, timestamp):
        """
        prev_c2w = prev_c2w.to(self.device).float()
        prev2_c2w_inv = get_past_c2w(-2).to(self.device).float().inverse()
        delta = prev_c2w @ prev2_c2w_inv
        estimated_new_c2w = delta @ prev_c2w
        """
        t0, pose0 = self.pose_hist[-2]
        t1, pose1 = self.pose_hist[-1]
        t = (timestamp - t0) / (t1 - t0)
        pose0 = pose0.to(self.device)
        pose1 = pose1.to(self.device)
        return pose_ops.pose_slerp(pose0, pose1, t)

    def copy_last(self):
        _, pose = self.pose_hist[-1]
        return pose
    
    def add_noise(self, pose, noise_scale):
        logger.debug(f'pose init before noise: {pose.cpu().numpy().round(4)}')
        has_batch_dim = pose.dim() == 2  # (B,7) or (7,)
        if not has_batch_dim:
            pose = pose.unsqueeze(0)  # (1,7)
        if noise_scale <= 0:
            return pose
        
        # lie_noise = lietorch.SE3.Random(pose.shape[0], sigma=float(noise_scale), device=self.device)
        # lie_pose = lietorch.SE3(pose.to(self.device))
        # lie_pose = lie_pose * lie_noise
        # pose = lie_pose.data
        noise = torch.rand(pose.shape[0], 6, device=self.device)
        noise = noise * 2 - 1
        noise = noise * noise_scale
        pose = lietorch.SE3(pose) * lietorch.SE3.exp(noise)
        pose = pose.data
        
        if not has_batch_dim:
            pose = pose[0]
        logger.debug(f'pose init after noise: {pose.cpu().numpy().round(4)}')
        return pose
