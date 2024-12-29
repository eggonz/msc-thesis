import os
import traceback
import datetime
import traceback
import torch
from collections import OrderedDict
from lietorch import SE3
from time import sleep
import torch.multiprocessing as mp
from src.droid_net import DroidNet
from src import conv_onet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.Renderer import Renderer
from src.utils.generate_mesh import generate_mesh_kf, generate_mesh_every
from src.utils.eval_recon import eval_recon_with_cfg
from src.common import setup_seed, update_cam
from src.tracker import Tracker
from src.mapper import Mapper
from src import config
from multiprocessing.managers import BaseManager
from src.neural_point import NeuralPointCloud
from src.droid_backend import DroidBackend
from src.utils.loggers import WandbLogger
from src.utils.datasets import get_dataset
from src.utils.pose_trajectory import PoseHistory, OutputTrajectory
from src.utils.eval_traj import kf_traj_eval, full_traj_eval, out_traj_eval
import numpy as np
from src.expert_dataset import ExpertDataSaver


class SLAM:
    def __init__(self, args, cfg, output_dir=None):
        super().__init__()
        
        if output_dir is None:
            timestamp = datetime.datetime.now().isoformat()
            output_dir = os.path.join(cfg["data"]["output"], cfg["expname"], cfg['setting'], cfg['scene'], timestamp)

        os.makedirs(output_dir, exist_ok=True)
        cfg['data']['output'] = output_dir
        print("Running SLAM in directory:", output_dir)

        self.args = args
        self.cfg = cfg

        self.device = args.device
        self.verbose = cfg['verbose']

        self.dataset = cfg['dataset']  # dataset name

        self.mode = cfg['mode']  # TODO force mode=RGBD
        self.only_tracking: bool = cfg['only_tracking']
        self.offline_mapping: bool = cfg['offline_mapping']
        self.make_video = args.make_video

        # Logs
        self.wandb_logger = WandbLogger(cfg) if cfg['wandb'] else None

        # Output path
        self.output = output_dir
        self.ckptsdir = os.path.join(self.output, 'ckpts')

        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'mesh'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'rendered_image'), exist_ok=True)

        config.save_config(cfg, f'{self.output}/cfg.yaml')

        # Update camera params
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        # Load pretrained models
        self.droid_net: DroidNet = DroidNet()
        self.conv_o_net: conv_onet.models.decoder.POINT = conv_onet.config.get_model(cfg)  # TODO convo to mapper
        self.load_pretrained(cfg, self.device)

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()
        self.tracking_finished = torch.zeros((1)).int()
        self.tracking_finished.share_memory_()

        #>>> for f2m-tracking TODO needed?
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_idx[0] = -1
        self.exposure_feat = torch.zeros((1, cfg['model']['exposure_dim'])).normal_(
            mean=0, std=0.01).to(self.device)
        self.exposure_feat.share_memory_()
        #<<<
        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(cfg, args)
        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.droid_net, video=self.video, device=self.device)
        
        BaseManager.register('NeuralPointCloud', NeuralPointCloud)
        BaseManager.register('ExpertDataSaver', ExpertDataSaver)
        manager = BaseManager()
        manager.start()
        self.npc:NeuralPointCloud = manager.NeuralPointCloud(self.cfg)
        self.expert_data_saver:ExpertDataSaver = manager.ExpertDataSaver()

        self.renderer:Renderer = Renderer(self.cfg, full_res=False)
        self.tracker: Tracker = ...
        self.mapper: Mapper = ...

        # shared pose estimation
        stream = get_dataset(cfg)
        n_img = len(stream)
        self.f2f_pose_history = PoseHistory(n_img, device=self.device)
        self.f2m_pose_history = PoseHistory(n_img, device=self.device)
        self.mix_pose_history = PoseHistory(n_img, device=self.device)
        self.gt_pose_history = PoseHistory(n_img, device=self.device)

    def load_pretrained(self, cfg, device):
        # Load DroidNet
        droid_pretrained = cfg['tracking']['pretrained']
        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(droid_pretrained).items()
        ])
        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.to(device).eval()  # frozen
        self.droid_net.share_memory()
        print(f'INFO: loaded DroidNet pretrained checkpoint from {droid_pretrained}!')

        # Load ConvONet
        convo_pretrained = cfg['mapping']['pretrained']  # middle_fine decoder
        convo_ckpt = torch.load(convo_pretrained, map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in convo_ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8 + 7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8 + 5:]
                    fine_dict[key] = val
        self.conv_o_net.geo_decoder.load_state_dict(middle_dict, strict=False)
        self.conv_o_net.to(device)
        self.conv_o_net.share_memory()
        print(f'INFO: loaded ConvONet pretrained checkpoint from {convo_pretrained}!')

    def tracking(self, rank, stream, pipe, pose_mixer):
        self.tracker = Tracker(self, pipe)
        print('Tracking Triggered!')
        self.all_trigered += 1

        os.makedirs(f'{self.output}/mono_priors/depths', exist_ok=True)
        os.makedirs(f'{self.output}/mono_priors/normals', exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        try:
            self.tracker.run(stream, pose_mixer)
        except Exception as e:
            traceback.print_exc()
            print('[ERR]', e)
            traceback.print_exc()
        self.tracking_finished += 1
        torch.cuda.empty_cache()
        print('Tracking Done!')
        if self.only_tracking:
            self.terminate(rank=-1, stream=stream)
    
    def mapping(self, rank, stream, pipe):
        if self.only_tracking:
            self.all_trigered += 1
            return
        self.mapper = Mapper(self, pipe)
        print('Mapping Triggered!')

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])
        
        if self.mapper.use_dynamic_radius:
            os.makedirs(f'{self.output}/dynamic_r_frame', exist_ok=True)
        if self.mapper.encode_exposure:
            os.makedirs(f"{self.output}/ckpts/color_decoder", exist_ok=True)
        
        while self.all_trigered < self.num_running_thread:
            pass
        config_name = os.path.basename(self.args.config)
        try:
            self.mapper.run()
        except Exception as e:
            traceback.print_exc()
            print('[ERR]', e)
            traceback.print_exc()
        torch.cuda.empty_cache()
        print('Mapping Done!')

        self.terminate(rank=-1, stream=stream)

    def backend(self):
        print("Backend Optimization Triggered!")
        from .droid_backend import DroidBackend
        self.ba = DroidBackend(self.droid_net,self.video,self.args,self.cfg)
        
        t = self.video.counter.value
        torch.cuda.empty_cache()
        print("#" * 32)
        self.ba.dense_ba(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.ba.dense_ba(12)

    def terminate(self, rank, stream=None):
        """ fill poses for non-keyframe images and evaluate """
        print("Terminate: Triggered!")
        
        if not self.offline_mapping:
            if self.cfg['tracking']['backend']['final_ba']:
                self.backend()
            # NOTE: ExpertSLAM does not need final refinement, since we use pose history directly
            # if not self.only_tracking:
                # print('Doing final refinement...')
                # self.mapper.final_refine(save_final_pcl=True)
                # print('Final refinement done!')
            if self.make_video:
                print('Saving video...')
                self.video.save_video(f"{self.output}/video.npz")
                print('Video saved!')

        torch.cuda.empty_cache()

        do_evaluation = self.make_video  # TODO allow control over this variable
        if do_evaluation:
            # NOTE: ExpertSLAM can ignore KF evaluation, we do it just for comparison
            # this generates metrics: kf_ate_rmse, pose_scale
            try:
                pe_statistics, traj_scale, r_a, t_a = kf_traj_eval(
                    f"{self.output}/video.npz",
                    f"{self.output}/traj",
                    "kf_traj",stream,self.wandb_logger)
            except Exception as e:
                traceback.print_exc()
                print('[ERR]', e)
                traceback.print_exc()

            # NOTE: ExpertSLAM can ignore full trajectory evaluation, we do it just for comparison
            # this generates metrics: full_ate_rmse
            # try:
            #     outtraj = OutputTrajectory.from_traj_filler(self.traj_filler, stream)
            #     full_traj, full_traj_aligned, full_traj_ref = full_traj_eval(outtraj,
            #                 f"{self.output}/traj",
            #                 "full_traj",
            #                 stream, self.wandb_logger)
            #     np.save(f"{self.output}/traj/full_traj_aligned.npy",full_traj_aligned.poses_se3)
            #     np.save(f"{self.output}/traj/full_traj_gt.npy",full_traj_ref.poses_se3)
            # except Exception as e:
            #     print('[ERR]', e)

            # this generates metrics: out_ate_rmse
            out_traj = OutputTrajectory.from_pose_history(self.mix_pose_history)
            _, out_traj_aligned, out_traj_ref = out_traj_eval(out_traj,
                        f"{self.output}/traj",
                        "out_traj",
                        stream, self.wandb_logger)
            np.save(f"{self.output}/traj/out_traj_aligned.npy",out_traj_aligned.poses_se3)
            np.save(f"{self.output}/traj/out_traj_gt.npy",out_traj_ref.poses_se3)

            if not self.only_tracking:

                # NOTE: ExpertSLAM can ignore KF evaluation
                # self.mapper.eval_kf_imgs(traj_scale)
                # generate_mesh_kf(f"{self.output}/cfg.yaml",rendered_path="rendered_every_keyframe",mesh_name_suffix="kf")
                
                # this generates metrics: avg_psnr, avg_ssim, avg_lpips + masked
                self.mapper.eval_imgs(None, out_traj)  # FIXME traj_scale unused
                generate_mesh_every(f"{self.output}/cfg.yaml", "out_traj_aligned.npy",  # NOTE: ExpertSLAM uses out_traj, not full_traj
                                    rendered_path="rendered_every_frame",mesh_name_suffix="every", timestamps=out_traj.get_trajectory()[0])
                
                if self.cfg["dataset"] in ["replica"]:
                    # this generates metrics: recall_every, precision_every, f-score_every, ...
                    try:
                        recon_result = eval_recon_with_cfg(f"{self.output}/cfg.yaml",
                                                           eval_3d=True,eval_2d=False, # FIXME eval2d not working
                                                           kf_mesh=False, every_mesh=True)
                        if self.wandb_logger:
                            self.wandb_logger.log(recon_result)
                        output_str = ""
                        for k, v in recon_result.items():
                            output_str += f"{k}: {v}\n"
                        out_path=f'{self.output}/logs/metrics_mesh.txt'
                        with open(out_path, 'w+') as fp:
                            fp.write(output_str)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        traceback.print_exc()
                        print('[ERR]', e)
                        traceback.print_exc()

        self.f2f_pose_history.save(f"{self.output}/f2f_pose_history.npz")
        self.f2m_pose_history.save(f"{self.output}/f2m_pose_history.npz")
        self.mix_pose_history.save(f"{self.output}/mix_pose_history.npz")
        self.gt_pose_history.save(f"{self.output}/gt_pose_history.npz")

        import shutil
        if os.path.exists(f'{self.output}/dynamic_r_frame'):
            shutil.rmtree(f'{self.output}/dynamic_r_frame')
        if os.path.exists(f'{self.output}/mono_priors'):
            shutil.rmtree(f'{self.output}/mono_priors')
        if os.path.exists(f'{self.output}/rendered_every_frame'):
            shutil.rmtree(f'{self.output}/rendered_every_frame')
        if os.path.exists(f'{self.output}/rendered_every_keyframe'):
            shutil.rmtree(f'{self.output}/rendered_every_keyframe')
        if not self.cfg['mapping']['save_ckpts']:
            if os.path.exists(f'{self.output}/ckpts'):
                shutil.rmtree(f'{self.output}/ckpts')

        print("Terminate: Done!")

    def run(self, stream, pose_mixer):  # TODO pass mixer in init
        """
        Run the SLAM system.

        stream: dataset
        """

        m_pipe, t_pipe = mp.Pipe()
        processes = {
            'tracking': mp.Process(target=self.tracking, args=(0,stream, t_pipe, pose_mixer)),
            'mapping': mp.Process(target=self.mapping, args=(1,stream, m_pipe)),
            # 'optimizing': mp.Process(target=self.optimizing, args=(2, stream, not dont_run)),
        }

        self.num_running_thread[0] += len(processes)
        for name, p in processes.items():
            p.start()
            print(f"Process '{name}' started with PID: {p.pid}")

        for name, p in processes.items():
            p.join()
            print(f"Process '{name}' joined with PID: {p.pid}")

        if self.wandb_logger:
            print('wandb finished.')
            self.wandb_logger.finish()

        torch.cuda.empty_cache()
