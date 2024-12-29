import os
import shutil
import traceback
import subprocess
import time
import cv2
import numpy as np
import open3d as o3d
import torch

import lietorch
from ast import literal_eval
from colorama import Fore, Style
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from multiprocessing.connection import Connection
from src.common import (get_camera_from_tensor, get_samples, get_samples_with_pixel_grad,
                        get_tensor_from_camera, random_select, project_point3d_to_image_batch,)
from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.Visualizer import Visualizer
from src.utils.Renderer import Renderer
from src.utils.eval_render import eval_kf_imgs, eval_imgs, EvalerKf, EvalerEvery
from src.neural_point import NeuralPointCloud, update_points_pos, proj_depth_map, get_droid_render_depth, select_points_from_video, select_points_first_frame
from src.depth_video import DepthVideo
from src import conv_onet
from src.utils.loggers import CkptLogger

import torchvision.transforms as Transforms
transform = Transforms.ToPILImage()
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim

import wandb
import functools
print = functools.partial(print,flush=True)

class Mapper(object):
    """
    Mapper thread.

    """
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.args = slam.args
        if self.cfg['only_tracking']:
            return
        self.pipe = pipe
        self.output = slam.output
        self.ckptsdir = slam.ckptsdir
        self.verbose = slam.verbose
        self.renderer:Renderer = slam.renderer
        self.video:DepthVideo = slam.video
        self.npc:NeuralPointCloud = slam.npc
        self.low_gpu_mem = True
        self.mapping_idx = slam.mapping_idx  # TODO need?
        self.device = self.cfg['mapping']['device']
        self.decoders:conv_onet.models.decoder.POINT = slam.conv_o_net
        

        # self.estimate_c2w_list = slam.estimate_c2w_list
        # self.estimate_wq_list = slam.estimate_wq_list
        # self.gt_c2w_list = slam.gt_c2w_list
        self.exposure_feat_shared = slam.exposure_feat
        self.exposure_feat = self.exposure_feat_shared[0].clone().requires_grad_()
        self.wandb_logger = slam.wandb_logger
        self.gt_camera = self.cfg['tracking']['gt_camera']  # TODO not working
        self.bind_npc_with_pose = self.cfg['pointcloud']['bind_npc_with_pose']
        if self.cfg["offline_mapping"]:
            self.bind_npc_with_pose = False
        if self.gt_camera:
            assert self.cfg['data']['use_gt_wq']
        self.use_gt_wq = self.cfg['data']['use_gt_wq']
        if self.use_gt_wq:
            assert self.cfg["offline_mapping"] 

        self.use_dynamic_radius = self.cfg['pointcloud']['use_dynamic_radius']
        self.dynamic_r_add, self.dynamic_r_query = None, None
        self.encode_exposure = self.cfg['model']['encode_exposure']
        self.radius_add_max = self.cfg['pointcloud']['radius_add_max']
        self.radius_add_min = self.cfg['pointcloud']['radius_add_min']
        self.radius_query_ratio = self.cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = self.cfg['pointcloud']['color_grad_threshold']
        self.fix_geo_decoder = self.cfg['mapping']['fix_geo_decoder']
        self.fix_color_decoder = self.cfg['mapping']['fix_color_decoder']
        self.eval_rec = self.cfg['meshing']['eval_rec']
        if self.gt_camera:
            self.cfg['mapping']['BA'] = False
        self.BA = self.cfg['mapping']['BA']
        self.BA_cam_lr = self.cfg['mapping']['BA_cam_lr']
        self.ckpt_freq = self.cfg['mapping']['ckpt_freq']
        self.mapping_pixels = self.cfg['mapping']['pixels']
        self.pixels_adding = self.cfg['mapping']['pixels_adding']
        self.pixels_based_on_color_grad = self.cfg['mapping']['pixels_based_on_color_grad']
        self.num_joint_iters = self.cfg['mapping']['iters']
        self.geo_iter_first = self.cfg['mapping']['geo_iter_first']
        self.iters_first = self.cfg['mapping']['iters_first']
        self.every_frame = self.cfg['mapping']['every_frame']  # TODO unused?
        self.w_color_loss = self.cfg['mapping']['w_color_loss']
        self.keyframe_every = self.cfg['mapping']['keyframe_every']
        self.geo_iter_ratio = self.cfg['mapping']['geo_iter_ratio']
        self.BA_iter_ratio = self.cfg['mapping']['BA_iter_ratio']
        self.vis_inside = self.cfg['mapping']['vis_inside']
        self.mapping_window_size = self.cfg['mapping']['mapping_window_size']
        self.frustum_feature_selection = self.cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = self.cfg['mapping']['keyframe_selection_method']
        self.frustum_edge = self.cfg['mapping']['frustum_edge']
        self.save_ckpts = self.cfg['mapping']['save_ckpts']
        self.save_rendered_image = self.cfg['mapping']['save_rendered_image']
        self.min_iter_ratio = self.cfg['mapping']['min_iter_ratio']

        self.pix_warping = self.cfg['mapping']['pix_warping']

        self.w_pix_warp_loss = self.cfg['mapping']['w_pix_warp_loss']
        self.w_geo_loss = self.cfg['mapping']['w_geo_loss']
        self.w_geo_loss_first_stage = self.cfg['mapping']['w_geo_loss_first_stage']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(self.cfg, device=self.device)
        self.n_img = len(self.frame_reader)
        self.ckpt_logger = CkptLogger(self)
        self.visualizer = Visualizer(freq=1,  # show all keyframes, ignore cfg.mapping.vis_freq
                                     inside_freq=self.cfg['mapping']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                     verbose=self.verbose, device=self.device, wandb_logger=self.wandb_logger,
                                     vis_inside=self.vis_inside, total_iters=self.num_joint_iters,
                                     img_dir=os.path.join(self.output, 'rendered_image'))
        # self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H_o, slam.W_o, slam.fx_o, slam.fy_o, slam.cx_o, slam.cy_o
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        self.render_depth = self.cfg['mapping']['render_depth']
        self.use_mono_to_complete = self.cfg['mapping']['use_mono_to_complete']
        self.init_idx = 0

        if self.cfg["mapping"]["save_depth"]:
            os.makedirs(f'{self.output}/semi_dense_depth/droid', exist_ok=True)
            os.makedirs(f'{self.output}/semi_dense_depth/project', exist_ok=True)

        #self.evaler_kf = EvalerKf(self)
        self.evaler_every = EvalerEvery(self)

    def get_mask_from_c2w(self, c2w, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            depth_np (numpy.array): depth image of current frame. for each (x,y)<->(width,height)

        Returns:
            mask (tensor): mask for selected optimizable feature.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        points = self.npc.cloud_pos().cpu().numpy().reshape(-1, 3)

        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # flip the x-axis such that the pixel space is u from the left to right, v top to bottom.
        # without the flipping of the x-axis, the image is assumed to be flipped horizontally.
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = self.frustum_edge  # crop here on width and height
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        return np.where(mask)[0].tolist()

    def keyframe_selection_overlap(self, gt_color, mono_depth, c2w, keyframe_dict, k, N_samples=8, pixels=200):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            mono_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, mono_depth, gt_color = get_samples(
            0, H, 0, W, pixels,
            fx, fy, cx, cy, c2w, mono_depth, gt_color, self.device, depth_filter=True)

        mono_depth = mono_depth.reshape(-1, 1)
        mono_depth = mono_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = mono_depth*0.8
        far = mono_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            if self.gt_camera:
                idx = keyframe['idx']
                c2w = np.copy(self.frame_reader.poses[idx])  # TODO using absolute poses
                c2w[:3, 1:3] *= -1
            else:
                video_idx = keyframe['video_idx']
                c2w = self.video.get_pose(video_idx,'cpu').numpy()  # relative to first pose
                c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            # flip the x-axis such that the pixel space is u from the left to right, v top to bottom.
            # without the flipping of the x-axis, the image is assumed to be flipped horizontally.
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def filter_point_before_add(self, rays_o, rays_d, gt_depth, prev_c2w):
        with torch.no_grad():
            points = rays_o[..., None, :] + \
                rays_d[..., None, :] * gt_depth[..., None, None]
            points = points.reshape(-1, 3).cpu().numpy()
            H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            if torch.is_tensor(prev_c2w):
                prev_c2w = prev_c2w.cpu().numpy()
            w2c = np.linalg.inv(prev_c2w)
            ones = np.ones_like(points[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [points, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)

            edge = 0
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
        return torch.from_numpy(~mask).to(self.device).reshape(-1)

    def get_c2w_and_depth(self,video_idx,idx,mono_depth,print_info=False):
        if idx == 0:
            c2w = self.video.get_first_pose()
            c2w[:3, 1:3] *= -1
            c2w = c2w.to(self.device)
            mono_depth_wq = mono_depth.clone()
            est_droid_depth = mono_depth.clone()  # droid not initialized yet
        elif self.gt_camera:
            raise NotImplementedError  # TODO use poses relative to idx=0
            gt_c2w = self.frame_reader.poses[idx].clone().to(self.device)  # absolute
            c2w[:3, 1:3] *= -1
            mono_depth_wq = mono_depth.clone()
            est_droid_depth = mono_depth.clone()
        else:
            est_droid_depth, valid_depth_mask, c2w, pose_scale = self.video.get_depth_and_pose(video_idx,self.device)
            if print_info:
                print(f"valid depth number: {valid_depth_mask.sum().item()}, " 
                      f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}")
            if valid_depth_mask.sum() < 100:
                print(f"Skip mapping frame {idx} because of not enough valid depth ({valid_depth_mask.sum()}).")                
                return None, None, None
            est_droid_depth[~valid_depth_mask] = 0
            c2w[:3, 1:3] *= -1
            # depth_weight = self.video.get_weight_offline(video_idx,self.device)
            if self.use_gt_wq:
                raise NotImplementedError

            mono_valid_mask = mono_depth < (mono_depth.mean()*3)
            valid_mask = mono_valid_mask*valid_depth_mask
            cur_wq = self.video.get_depth_scale_and_shift(video_idx,mono_depth[valid_mask],est_droid_depth[valid_mask], None)
            c2w = c2w.to(self.device)
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]
            # mono_depth_wq[~mono_valid_mask] = 0
        return c2w, mono_depth_wq, est_droid_depth
    
    def anchor_points(self, anchor_depth, anchor_full_pcl_depth, cur_gt_color, cur_c2w, cur_idx, cur_video_idx, init=False):
        edge = 0
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        anchor_mask = anchor_depth>0
        gt_color = cur_gt_color.to(self.device)
        # if init:
        #     add_pts_num = torch.clamp(self.pixels_adding * ((anchor_depth.median()/2.5)**2),
        #                               min=self.pixels_adding, max=self.pixels_adding*3).int().item()
        # else:
        add_pts_num = self.pixels_adding

        batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color, i, j = get_samples(
            edge, H-edge, edge, W-edge, add_pts_num,
            fx, fy, cx, cy, cur_c2w, anchor_depth, gt_color, self.device, depth_filter=True, return_index=True,mask=anchor_mask)

        frame_pts_add = 0
        if cur_idx == 0:
            # video disps not defined yet
            pts_gt, mask = select_points_first_frame(self.video, anchor_depth, self.npc.get_device())
        else:
            pts_gt, mask = select_points_from_video(self.video, cur_video_idx, self.npc.get_device())
        _ = self.npc.add_points(cur_video_idx, pts_gt, mask)
        _ = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color,
                                        cur_video_idx, i,j,
                                        dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None)
        print(f'{_} locations to add points.')
        frame_pts_add += _

        if self.pixels_based_on_color_grad > 0:

            batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color, i, j = get_samples_with_pixel_grad(
                edge, H-edge, edge, W-edge, self.pixels_based_on_color_grad,
                H, W, fx, fy, cx, cy, cur_c2w, anchor_depth, gt_color, self.device,
                anchor_mask,
                depth_filter=True, return_index=True)
            _ = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_anchor_depth, batch_gt_color,
                                            cur_video_idx, i,j,
                                            is_pts_grad=True, dynamic_radius=self.dynamic_r_add[j, i] if self.use_dynamic_radius else None)
            print(f'{_} locations to add points based on pixel gradients.')
            frame_pts_add += _
        return frame_pts_add

    def optimize_map(self, num_joint_iters, cur_idx, cur_depth, cur_gt_color, cur_gt_depth, cur_mono_depth,
                     cur_droid_depth,frame_pts_add,
                     keyframe_dict, keyframe_list, cur_c2w, init, color_refine=False):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            idx (int): the index/timestamp of current frame
            cur_depth (): TODO
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (): TODO
            cur_mono_depth (tensor): mono_depth image of the current camera.
            cur_droid_depth (): TODO
            frame_pts_add (): TODO
            [ gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame. ]
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 
            [ gt_wq (tensor 1x2): groundtruth scale and shift for estimated depth ]
            [ cur_wq (tensor 1x2): Current estimate of scale and shift ]
            color_refine (bool): whether to do color refinement (optimize color features with fixed color decoder).

        Returns:
            [ cur_mono_depth (tensor): return w*detph + q ]
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        edge = 0
        npc = self.npc
        cfg = self.cfg
        device = self.device
        bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4) 
        cur_r_query = self.dynamic_r_query/3.0*cur_depth
        # mean_r_query = cur_r_query.mean()
        # cur_r_query[cur_r_query==0] = mean_r_query 

        cur_mask  = cur_depth > 0       
        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_depth, cur_c2w, keyframe_dict[:-1], num)        

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        optimize_frame_dict = []
        print("Projecting pointcloud to keyframes ...")
        for frame in optimize_frame:
            if frame != -1:
                mono_depth = keyframe_dict[frame]['mono_depth'].to(device)
                gt_color = keyframe_dict[frame]['color'].to(device)
                video_idx = keyframe_dict[frame]['video_idx']
                idx = keyframe_dict[frame]['idx']
                c2w,mono_depth, droid_depth = self.get_c2w_and_depth(video_idx,idx,mono_depth)
                if c2w is None:
                    continue
                if self.cfg["mapping"]["render_depth"] == "droid":
                    render_depth = get_droid_render_depth(self.npc, self.cfg, c2w.clone(),
                                                           droid_depth, mono_depth, 
                                                           device,use_mono_to_complete=self.use_mono_to_complete)
                    render_mask = render_depth > 0
                    geo_loss_mask = droid_depth > 0
                elif self.cfg["mapping"]["render_depth"] == "mono":
                    render_depth = mono_depth
                    render_mask = torch.ones_like(mono_depth,dtype=torch.bool,device=mono_depth.device)
                    geo_loss_mask = render_mask.clone()
            else:
                if color_refine:
                    continue
                render_depth = cur_depth
                render_mask = cur_mask
                gt_color = cur_gt_color.to(device)
                c2w = cur_c2w
                geo_loss_mask = cur_droid_depth>0
            optimize_frame_dict.append({"frame":frame, "render_depth":render_depth, 
                                        "render_mask":render_mask, "gt_color":gt_color,"c2w":c2w,
                                        "geo_loss_mask":geo_loss_mask})

        # if self.save_selected_keyframes_info:
        #     keyframes_info = []
        #     for id, frame in enumerate(optimize_frame):
        #         if frame != -1:
        #             frame_idx = keyframe_list[frame]
        #             tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
        #             tmp_est_c2w = keyframe_dict[frame]['est_c2w']
        #         else:
        #             frame_idx = idx
        #             tmp_gt_c2w = gt_cur_c2w
        #             tmp_est_c2w = cur_c2w
        #         keyframes_info.append(
        #             {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
        #     self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        color_pcl_para = []
        geo_pcl_para = []
        depth_np = cur_depth.cpu().numpy()
        cur_depth = cur_depth.to(device)
        
        # clone all point feature from shared npc, (N_points, c_dim)
        npc_geo_feats = self.npc.get_geo_feats()
        npc_col_feats = self.npc.get_col_feats()
        self.cloud_pos_tensor = self.npc.cloud_pos()
        if self.encode_exposure:
            self.exposure_feat = self.exposure_feat_shared[0].clone(
            ).requires_grad_()

        if self.frustum_feature_selection:  # required if not color_refine
            masked_c_grad = {}
            mask_c2w = cur_c2w
            indices = self.get_mask_from_c2w(mask_c2w, depth_np)
            geo_pcl_grad = npc_geo_feats[indices].requires_grad_(True)
            color_pcl_grad = npc_col_feats[indices].requires_grad_(True)

            geo_pcl_para = [geo_pcl_grad]
            color_pcl_para = [color_pcl_grad]

            masked_c_grad['geo_pcl_grad'] = geo_pcl_grad
            masked_c_grad['color_pcl_grad'] = color_pcl_grad
            masked_c_grad['indices'] = indices
        else:
            masked_c_grad = {}
            geo_pcl_grad = npc_geo_feats.requires_grad_(True)
            color_pcl_grad = npc_col_feats.requires_grad_(True)

            geo_pcl_para = [geo_pcl_grad]
            color_pcl_para = [color_pcl_grad]

            masked_c_grad['geo_pcl_grad'] = geo_pcl_grad
            masked_c_grad['color_pcl_grad'] = color_pcl_grad

        if not self.fix_geo_decoder:
            decoders_para_list += list(
                self.decoders.geo_decoder.parameters())
        if not self.fix_color_decoder:
            decoders_para_list += list(
                self.decoders.color_decoder.parameters())

        if self.BA:
            camera_tensor_list = []  # list of [quad,T]
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w'].to(self.device)
                    else:
                        c2w = cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)  # [quad,T]
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)

        optim_para_list = [{'params': decoders_para_list, 'lr': 0},
                           {'params': geo_pcl_para, 'lr': 0},
                           {'params': color_pcl_para, 'lr': 0}]
        if self.BA:
            optim_para_list.append({'params': camera_tensor_list, 'lr': 0})
        if self.encode_exposure:
            optim_para_list.append(
                {'params': self.exposure_feat, 'lr': 0.001})
        optimizer = torch.optim.Adam(optim_para_list)

        if not init and not color_refine:
            num_joint_iters = np.clip(int(num_joint_iters*frame_pts_add/300), int(
                self.min_iter_ratio*num_joint_iters), 2*num_joint_iters)

        for joint_iter in range(num_joint_iters):
            tic = time.perf_counter()
            if self.frustum_feature_selection:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                indices = masked_c_grad['indices']
                npc_geo_feats[indices] = geo_feats
                npc_col_feats[indices] = col_feats
            else:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                npc_geo_feats = geo_feats  # all feats
                npc_col_feats = col_feats

            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters*self.geo_iter_ratio)):
                self.stage = 'geometry'
            else:
                self.stage = 'color'
            cur_stage = 'init' if init else 'stage'
            optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage][self.stage]['decoders_lr']
            optimizer.param_groups[1]['lr'] = cfg['mapping'][cur_stage][self.stage]['geometry_lr']
            if color_refine:
                optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage]['color']['decoders_lr']
                optimizer.param_groups[1]['lr'] = cfg['mapping'][cur_stage]['color']['geometry_lr']#/10.0
                optimizer.param_groups[2]['lr'] = cfg['mapping'][cur_stage]['color']['color_lr']#/10.0
            else:
                optimizer.param_groups[2]['lr'] = cfg['mapping'][cur_stage][self.stage]['color_lr']

            if self.BA:
                # when to conduct BA
                if joint_iter >= num_joint_iters*self.BA_iter_ratio:
                    optimizer.param_groups[3]['lr'] = self.BA_cam_lr
                else:
                    optimizer.param_groups[3]['lr'] = 0.0

            # cur_render_depth = cur_mono_depth
            # if self.vis_inside:
            #     self.visualizer.vis(idx, joint_iter, cur_gt_depth, cur_render_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
            #                         npc_geo_feats, npc_col_feats, freq_override=False,
            #                         dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos_tensor,
            #                         exposure_feat=self.exposure_feat)

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_render_depth_list = []
            batch_gt_color_list = []
            batch_r_query_list = []
            exposure_feat_list = []
            c2w_list = []
            img_gt_color_list = []
            img_render_depth_list = []
            indices_tensor = []  # used to index the predicted color from the decoder to
            # match with the per frame exposure feature
            frame_indexs = []
            geo_loss_mask_list = []

            camera_tensor_id = 0
            for frame_dict in optimize_frame_dict:
                frame = frame_dict["frame"]
                render_depth = frame_dict["render_depth"]
                render_mask = frame_dict["render_mask"]
                c2w = frame_dict["c2w"]
                gt_color = frame_dict["gt_color"]
                geo_loss_mask = frame_dict["geo_loss_mask"]

                batch_rays_o, batch_rays_d, batch_render_depth, batch_gt_color, i, j = get_samples(
                    edge, H-edge, edge, W-edge, pixs_per_image, 
                    fx, fy, cx, cy, c2w, render_depth, gt_color, self.device, 
                    depth_filter=True, 
                    return_index=True, mask=render_mask)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_render_depth_list.append(batch_render_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())
                geo_loss_mask_list.append(geo_loss_mask[j,i])
                    
                if self.pix_warping:
                    if c2w.shape[0]==4:
                        c2w_list.append(c2w)
                    elif c2w.shape[0]==3:
                        c2w_homo = torch.cat([c2w, bottom], dim=0)
                        c2w_list.append(c2w_homo)
                    else:
                        raise NotImplementedError
           
                    img_gt_color_list.append(gt_color)
                    img_render_depth_list.append(render_depth)

                if self.use_dynamic_radius:
                    if frame == -1:
                        batch_r_query_list.append(cur_r_query[j, i])
                    else:
                        r_query = keyframe_dict[frame]['dynamic_r_query']/3.0*render_depth
                        # mean_r_query = cur_r_query.mean()
                        # r_query[cur_r_query==0] = mean_r_query 
                        batch_r_query_list.append(r_query[j, i])

                if self.encode_exposure:  # needs to render frame by frame
                    exposure_feat_list.append(
                        self.exposure_feat if frame == -1 else keyframe_dict[frame]['exposure_feat'].to(device))
                
                # log frame idx of pixels
                frame_indices = torch.full(
                    (i.shape[0],), frame, dtype=torch.long, device=self.device)
                indices_tensor.append(frame_indices)
                frame_indexs.append(frame)

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_render_depth = torch.cat(batch_render_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)
            batch_geo_loss_mask = torch.cat(geo_loss_mask_list)

            if self.pix_warping:
                img_gt_colors = torch.stack(img_gt_color_list).to(self.device)
                c2ws = torch.stack(c2w_list, dim=0)

            r_query_list = torch.cat(
                batch_r_query_list) if self.use_dynamic_radius else None

            with torch.no_grad():
                if batch_render_depth.size(dim=0) == 0:
                    raise ValueError("Empty batch_render_depth not handled")
                inside_mask = batch_render_depth <= torch.minimum(
                    10*batch_render_depth.median(), 1.2*torch.max(batch_render_depth))

            batch_rays_d, batch_rays_o = batch_rays_d[inside_mask], batch_rays_o[inside_mask]
            batch_render_depth, batch_gt_color = batch_render_depth[inside_mask], batch_gt_color[inside_mask]
            batch_geo_loss_mask = batch_geo_loss_mask[inside_mask]

            if self.use_dynamic_radius:
                r_query_list = r_query_list[inside_mask]
            ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o, device, self.stage,
                                                 gt_depth=batch_render_depth, npc_geo_feats=npc_geo_feats,
                                                 npc_col_feats=npc_col_feats,
                                                 is_tracker=True if self.BA else False,
                                                 cloud_pos=self.cloud_pos_tensor,
                                                 dynamic_r_query=r_query_list,
                                                 exposure_feat=None)
            depth, uncertainty, color, valid_ray_mask, valid_ray_count = ret


            # depth_mask = (batch_render_depth > 0) & valid_ray_mask  # TODO why commented?
            depth_mask = (batch_render_depth > 0)
            depth_mask = depth_mask & (~torch.isnan(depth))
            # valid_geo_loss_mask = depth_mask * batch_geo_loss_mask
            valid_geo_loss_mask = depth_mask
            geo_loss = torch.abs(
                batch_render_depth[valid_geo_loss_mask]-depth[valid_geo_loss_mask]).sum()
            if self.stage == 'color':
                w_geo_loss = self.w_geo_loss
            else:
                w_geo_loss = self.w_geo_loss_first_stage
            loss = geo_loss*w_geo_loss
            
            indices_tensor = torch.cat(indices_tensor, dim=0)[inside_mask]
            frame_indexs = torch.tensor(frame_indexs).long().to(self.device)

            if self.stage == 'color':
                if self.encode_exposure:

                    start_end = []
                    for i in torch.unique_consecutive(indices_tensor, return_counts=False):
                        match_indices = torch.where(indices_tensor == i)[0]
                        start_idx = match_indices[0]
                        end_idx = match_indices[-1] + 1
                        start_end.append((start_idx.item(), end_idx.item()))
                    for i, exposure_feat in enumerate(exposure_feat_list):
                        start, end = start_end[i]
                        affine_tensor = self.decoders.color_decoder.mlp_exposure(
                            exposure_feat)
                        rot, trans = affine_tensor[:9].reshape(
                            3, 3), affine_tensor[-3:]
                        color_slice = color[start:end].clone()
                        color_slice = torch.matmul(color_slice, rot) + trans
                        color[start:end] = color_slice
                    color = torch.sigmoid(color)
                

                color_loss = torch.abs(batch_gt_color[depth_mask] - color[depth_mask]).sum()

                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss
            
            if self.pix_warping:
                pix_warping_edge = 5
                pixel_3d_pts = (batch_rays_o + batch_rays_d * depth[:,None])
                pixel_3d_pts = pixel_3d_pts.float()
                uv, z = project_point3d_to_image_batch(
                    c2ws, pixel_3d_pts.view(-1, 3, 1), fx, fy, cx, cy, self.device)

                uv = uv.view(1, pixel_3d_pts.shape[0], c2ws.shape[0], 2)  # [1, pn, Cn, 2]

                mask = (
                    (uv[0,:, :, 0] < W - pix_warping_edge)
                    * (uv[0,:, :, 0] > pix_warping_edge)
                    * (uv[0,:, :, 1] < H - pix_warping_edge)
                    * (uv[0,:, :, 1] > pix_warping_edge)
                )  # [Pn, Cn]

                mask = mask & (z.view(pixel_3d_pts.shape[0], c2ws.shape[0], 1)[:, :, 0] < 0)
                mask = mask & (frame_indexs[None, :] != indices_tensor[:, None])
                mask[mask.sum(dim=1) < 4] = False

                windows_reproj_idx = uv.permute(2, 1, 0, 3)  # Cn, pn, 1, 2
                windows_reproj_idx[..., 0] = windows_reproj_idx[..., 0] / W * 2.0 - 1.0
                windows_reproj_idx[..., 1] = windows_reproj_idx[..., 1] / H * 2.0 - 1.0


                # img_gt_colors [cn,height,width,3]    
                windows_reproj_gt_color = torch.nn.functional.grid_sample(
                    img_gt_colors.permute(0, 3, 1, 2).float(),              
                    windows_reproj_idx,
                    padding_mode="border",
                    align_corners=False
                ).permute(2, 0, 3, 1)  # [Pn, cn, 1, 3]


                tmp_windows_reproj_gt_color = windows_reproj_gt_color[:,:,0,:]
                tmp_batch_gt_color = batch_gt_color

                forward_reproj_loss = torch.nn.functional.smooth_l1_loss(tmp_windows_reproj_gt_color[mask],
                            tmp_batch_gt_color.unsqueeze(1).repeat(1,c2ws.shape[0],1)[mask], beta=0.1) * 1.0

                loss += self.w_pix_warp_loss* (forward_reproj_loss.sum())

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated params back to npc
            if self.frustum_feature_selection:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                indices = masked_c_grad['indices']
                npc_geo_feats, npc_col_feats = npc_geo_feats.detach(), npc_col_feats.detach()
                npc_geo_feats[indices], npc_col_feats[indices] = geo_feats.detach(
                ), col_feats.detach()
            else:
                geo_feats, col_feats = masked_c_grad['geo_pcl_grad'], masked_c_grad['color_pcl_grad']
                npc_geo_feats, npc_col_feats = geo_feats.detach(), col_feats.detach()

            toc = time.perf_counter()
            if self.wandb_logger is None:
                if joint_iter % 100 == 0:
                    if self.stage == 'geometry':
                        print('iter: ', joint_iter, ', time',
                              f'{toc - tic:0.6f}', ', geo_loss: ', f'{geo_loss.item():0.6f}')
                    else:
                        print('iter: ', joint_iter, ', time', f'{toc - tic:0.6f}',
                              ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')
                    if self.pix_warping:
                        print('      pix_warp_loss: ',f'{forward_reproj_loss.sum().item():0.6f}')
                    

            if joint_iter == num_joint_iters-1:
                print('idx: ', cur_idx, ', time', f'{toc - tic:0.6f}', ', geo_loss_pixel: ', f'{(geo_loss.item()/valid_geo_loss_mask.sum().item()):0.6f}',
                      ', color_loss_pixel: ', f'{(color_loss.item()/depth_mask.sum().item()):0.4f}')
                if self.wandb_logger:
                    self.wandb_logger.log({'time': float(f'{toc - tic:0.6f}'),
                                'geo_loss_pixel': float(f'{(geo_loss.item()/valid_geo_loss_mask.sum().item()):0.6f}'),
                                'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                'pts_total': self.npc.index_ntotal(),
                                'num_joint_iters': num_joint_iters})

        if (not self.vis_inside) or init:
            self.visualizer.vis(cur_idx, num_joint_iters-1, cur_gt_depth, 
                                cur_depth, cur_droid_depth, cur_mono_depth,
                                cur_gt_color, cur_c2w,
                                self.npc, self.decoders,
                                npc_geo_feats, npc_col_feats, self.cfg,
                                freq_override=init,
                                dynamic_r_query=cur_r_query,
                                cloud_pos=self.cloud_pos_tensor, exposure_feat=self.exposure_feat,
                                cur_total_iters=num_joint_iters, save_rendered_image=self.save_rendered_image)
        
        if self.frustum_feature_selection:
            self.npc.update_geo_feats(geo_feats, indices=indices)
            self.npc.update_col_feats(col_feats, indices=indices)
        else:
            self.npc.update_geo_feats(npc_geo_feats.detach().clone())
            self.npc.update_col_feats(npc_col_feats.detach().clone())

        # self.npc_geo_feats = npc_geo_feats
        # self.npc_col_feats = npc_col_feats
        print('Mapper has updated point features.')

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())  # [quad,T] -> 4x4 matrix
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())  # [quad,T] -> 4x4 matrix
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.encode_exposure:
            self.exposure_feat_all.append(self.exposure_feat.detach().cpu())
            torch.save(self.decoders.color_decoder.state_dict(),
                       f'{self.output}/ckpts/color_decoder/{cur_idx:05}.pt')

        if self.BA:
            return cur_c2w
        else:
            return None

    def mapping_keyframe(self, idx, video_idx, mono_depth, 
                         outer_joint_iters, num_joint_iters,
                         gt_color, gt_depth, init=False, color_refine=False):
        if self.bind_npc_with_pose:
            print("Updating pointcloud position ...")
            update_points_pos(self.npc, self.video, self.frame_reader)

        cur_c2w,depth_wq, droid_depth = self.get_c2w_and_depth(video_idx,idx,mono_depth,print_info=True)
        if cur_c2w is None:
            return False

        if self.cfg["mapping"]["render_depth"] == "droid":
            anchor_depth = droid_depth.clone()
            anchor_depth_invalid = (anchor_depth==0)
            anchor_depth[anchor_depth_invalid] = depth_wq[anchor_depth_invalid]
            anchor_full_pcl_depth = droid_depth.clone()
        elif self.cfg["mapping"]["render_depth"] == "mono":
            anchor_depth = depth_wq.clone()
            anchor_full_pcl_depth = depth_wq.clone()
        self.dynamic_r_add = self.dynamic_r_add/3.0 * anchor_depth

        frame_pts_add = 0
        if not color_refine:
            frame_pts_add = self.anchor_points(anchor_depth, anchor_full_pcl_depth, gt_color, cur_c2w, idx, video_idx, init)
        
        if self.cfg["mapping"]["render_depth"] == "droid":
            render_depth = get_droid_render_depth(self.npc, self.cfg, cur_c2w.clone(), 
                                                  droid_depth, depth_wq, self.device, idx, 
                                                  use_mono_to_complete=self.use_mono_to_complete)
        elif self.cfg["mapping"]["render_depth"] == "mono":
            render_depth = depth_wq

        if color_refine:
            self.dynamic_r_query = torch.load(f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt', map_location=self.device)

        for outer_joint_iter in range(outer_joint_iters):
            # start BA when having enough keyframes
            self.BA = (len(self.keyframe_list) >
                        4) and self.cfg['mapping']['BA']
                            
            c2w_from_opt = self.optimize_map(num_joint_iters, idx, render_depth, gt_color, gt_depth,
                                    depth_wq, droid_depth, frame_pts_add,
                                    self.keyframe_dict, self.keyframe_list, cur_c2w, init, 
                                    color_refine=color_refine)
            if self.BA:
                cur_c2w = c2w_from_opt
                # self.estimate_c2w_list[idx] = cur_c2w
        return True
    
    def run(self):
        cfg = self.cfg
        self.exposure_feat_all = [] if self.encode_exposure else None

        init = True
        while (1):
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp']
            video_idx = frame_info['video_idx']
            is_finished = frame_info['end']
            if is_finished:
                break

            # >>> Map keyframe
            if self.verbose:
                print(Fore.GREEN)
                print(">>> Mapping Frame ", idx)
                print(Style.RESET_ALL)
            
            # timestamp, image, gt_depth, intrinsics, gt_c2w, gt_color_full, gt_depth_full = self.frame_reader[idx]
            _, image, gt_depth, _, _, gt_color_full, gt_depth_full = self.frame_reader[idx]
            # mono_depth_input = load_mono_depth(idx,self.cfg)
            mono_depth_input = gt_depth.clone()
            
            gt_color = image.to(self.device).squeeze(0).permute(1,2,0)
            gt_depth = gt_depth.to(self.device)
            # gt_color = gt_color_full.to(self.device)
            # gt_depth = gt_depth_full.to(self.device)
            mono_depth = mono_depth_input.to(self.device)

            if self.use_dynamic_radius:
                ratio = self.radius_query_ratio
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)  # range 0~1
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                        self.radius_add_max, self.radius_add_max, self.radius_add_min])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                          ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
                dynamic_r_add = fn_map_r_add(color_grad_mag)
                dynamic_r_query = fn_map_r_query(color_grad_mag)
                self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                    self.device), torch.from_numpy(dynamic_r_query).to(self.device)
                torch.save(
                        self.dynamic_r_query, f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt')

            outer_joint_iters = 1
            if not init:
                num_joint_iters = cfg['mapping']['iters']
                self.mapping_window_size = cfg['mapping']['mapping_window_size']*(
                    2 if self.n_img > 4000 else 1)
            else:
                self.init_idx = idx
                num_joint_iters = self.iters_first  # more iters on first run

            print(f"m start: allocated {torch.cuda.memory_allocated()/(1024*1024*1024.0)}GB, "
                  f"reserved {torch.cuda.memory_reserved()/(1024*1024*1024.0)}GB")
            valid = self.mapping_keyframe(idx,video_idx,mono_depth,outer_joint_iters,num_joint_iters,
                                          gt_color,gt_depth,init,color_refine=False)
            torch.cuda.empty_cache()
            print(f"m end: allocated {torch.cuda.memory_allocated()/(1024*1024*1024.0)}GB, "
                  f"reserved {torch.cuda.memory_reserved()/(1024*1024*1024.0)}GB")        
            
            init = False
            if not valid:
                self.pipe.send("continue")
                continue


            if self.verbose:
                print(Fore.YELLOW)
                print("<<< Mapped Frame ", idx)
                print(Style.RESET_ALL)
            # <<< Map keyframe

            # >>> Evaluation

            #metrics = self.evaler_kf.process_frame(idx, video_idx)

            c2w_pose = self.video.get_pose_tensor(video_idx).detach().clone()
            c2w_matrix = lietorch.SE3(c2w_pose).matrix().data.cpu().numpy()
            metrics = self.evaler_every.process_frame(idx, est_c2w=c2w_matrix)
            metrics = {f'{k}_kf': v for k, v in metrics.items()}

            if self.wandb_logger:
                #self.wandb_logger.log({'psnr_frame': metrics['masked_psnr']})
                self.wandb_logger.log(metrics)

            # <<< Evaluation

            self.keyframe_list.append(idx)
            dic_of_cur_frame = {'idx': idx, 'color': gt_color.detach().cpu(),
                                'video_idx': video_idx,
                                'mono_depth': mono_depth_input.detach().clone().cpu(),
                                'gt_depth': gt_depth.detach().cpu()}
            
            if self.use_dynamic_radius:
                dic_of_cur_frame.update(
                    {'dynamic_r_query': self.dynamic_r_query.detach()})
            if self.encode_exposure:
                dic_of_cur_frame.update(
                    {'exposure_feat': self.exposure_feat.detach().cpu()})
            self.keyframe_dict.append(dic_of_cur_frame)

            if (video_idx % 3 == 0):  # TODO tmp debug logging
                cloud_pos = np.array(self.npc.input_pos().cpu())
                cloud_rgb = np.array(self.npc.input_rgb().cpu())
                point_cloud = np.hstack((cloud_pos, cloud_rgb))
                if self.wandb_logger:
                    self.wandb_logger.log(
                        {f'Cloud/point_cloud_{idx:05d}': wandb.Object3D(point_cloud)})
                else:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud_pos)
                    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)
                    o3d.io.write_point_cloud(
                        f'{self.output}/mapping_vis/point_cloud_{idx:05d}.ply', pcd)

            # FIXME CONTINUE HERE adapt logger inputs from Point to Expert
            if (idx > 0 and idx % self.ckpt_freq == 0) or idx == self.n_img-1:  # TODO tmp debug ckpt
                self.ckpt_logger.log(
                    idx,
                    self.keyframe_dict,
                    self.keyframe_list,
                    npc=self.npc,
                    exposure_feat=self.exposure_feat_all if self.encode_exposure else None,
                )

            self.mapping_idx[0] = idx
            torch.cuda.empty_cache()
            self.pipe.send("continue")
                    
        # Mesh the rendered color and depth images and evaluate the mesh
        # self.eval_reconstructions()

    def final_refine(self,save_final_pcl=True):
        video_idx = self.video.counter.value-1
        idx = int(self.video.timestamp[video_idx])
        num_joint_iters = self.cfg['mapping']['iters']
        self.mapping_window_size = self.video.counter.value-1  # use all keyframes
        outer_joint_iters = 5
        self.geo_iter_ratio = 0.0
        num_joint_iters *= 2
        self.fix_color_decoder = True
        self.frustum_feature_selection = False
        self.keyframe_selection_method = 'global'
        # timestamp, image, gt_depth, intrinsics, gt_c2w, gt_color_full, gt_depth_full = self.frame_reader[idx]
        _, image, gt_depth, _, _, gt_color_full, gt_depth_full = self.frame_reader[idx]
        # mono_depth = load_mono_depth(idx,self.cfg)
        mono_depth = gt_depth.clone()

        gt_color = image.to(self.device).squeeze(0).permute(1,2,0)
        gt_depth = gt_depth.to(self.device)
        # gt_color = gt_color_full.to(self.device)
        # gt_depth = gt_depth_full.to(self.device)
        mono_depth = mono_depth.to(self.device)
        self.mapping_keyframe(idx,video_idx,mono_depth,outer_joint_iters,num_joint_iters,
                              gt_color,gt_depth,init=False,color_refine=True)

        if save_final_pcl:
            cloud_pos = self.npc.input_pos().cpu().numpy()
            cloud_rgb = self.npc.input_rgb().cpu().numpy()
            point_cloud = np.hstack((cloud_pos, cloud_rgb))
            npc_cloud = self.npc.cloud_pos().cpu().numpy()
            np.save(f'{self.output}/final_point_cloud', point_cloud)
            np.save(f'{self.output}/npc_cloud', npc_cloud)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_pos)
            pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)
            o3d.io.write_point_cloud(
                f'{self.output}/final_point_cloud.ply', pcd)
            print('Saved point cloud and point normals.')
            if self.wandb_logger:
                self.wandb_logger.log(
                    {f'Cloud/point_cloud_{idx:05d}': wandb.Object3D(point_cloud)})
        if self.low_gpu_mem:
            torch.cuda.empty_cache()


Mapper.eval_kf_imgs = eval_kf_imgs  # TODO change to pure functions instead of mapper methods
Mapper.eval_imgs = eval_imgs