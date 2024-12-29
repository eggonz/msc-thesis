import numpy as np
import os
import torch
import lietorch
from colorama import Fore, Style
from scipy.interpolate import interp1d
from skimage import filters
from skimage.color import rgb2gray
from torch.autograd import Variable

from src.common import (
    get_camera_from_tensor,
    get_samples,
    get_rays_from_uv,
    get_tensor_from_camera,
    get_selected_index_with_grad,
    update_cam,
)
from src.utils.Visualizer import Visualizer
from src.utils.loggers import logger
from src.utils.Renderer import Renderer


class F2MTracker:
    def __init__(self, slam):
        self.device = slam.device
        self.output = slam.output
        self.verbose = slam.verbose
        self.wandb_logger = slam.wandb_logger

        self.dynamic_r_add = None
        self.dynamic_r_query = None

        self.use_dynamic_radius = slam.cfg['pointcloud']['use_dynamic_radius']
        self.radius_query_ratio = slam.cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = slam.cfg['pointcloud']['color_grad_threshold']
        self.radius_add_max = slam.cfg['pointcloud']['radius_add_max']
        self.radius_add_min = slam.cfg['pointcloud']['radius_add_min']

        self.sample_with_color_grad = slam.cfg['tracking']['f2m']['sample_with_color_grad']
        
        self.sample_with_color_grad = slam.cfg['tracking']['f2m']['sample_with_color_grad']
        self.ignore_edge_W = slam.cfg['tracking']['f2m']['ignore_edge_W']
        self.ignore_edge_H = slam.cfg['tracking']['f2m']['ignore_edge_H']
        self.use_color_in_tracking = slam.cfg['tracking']['f2m']['use_color_in_tracking']  # must be True for RGB-only refinements
        self.handle_dynamic = slam.cfg['tracking']['f2m']['handle_dynamic']
        self.depth_limit = slam.cfg['tracking']['f2m']['depth_limit']

        self.sample_with_color_grad = slam.cfg['tracking']['f2m']['sample_with_color_grad']
        self.w_color_loss = slam.cfg['tracking']['f2m']['w_color_loss']
        self.separate_LR = slam.cfg['tracking']['f2m']['separate_LR']
        self.encode_exposure = slam.cfg['model']['encode_exposure']  # False

        self.cam_lr = slam.cfg['tracking']['f2m']['lr']
        self.tracking_pixels = slam.cfg['tracking']['f2m']['pixels']
        self.num_cam_iters = slam.cfg['tracking']['f2m']['iters']

        logger.debug(f"Original camera params: H={slam.cfg['cam']['H']} W={slam.cfg['cam']['W']} fx={slam.cfg['cam']['fx']} fy={slam.cfg['cam']['fy']} cx={slam.cfg['cam']['cx']} cy={slam.cfg['cam']['cy']}")
        logger.debug(f" Updated camera params: H={slam.H} W={slam.W} fx={slam.fx} fy={slam.fy} cx={slam.cx} cy={slam.cy}")

        # NOTE: change with f2m-resolution
        # Use original camera params instead of modified ones
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(slam.cfg, full_res=False)
        self.renderer = Renderer(slam.cfg, full_res=False)

        self.vis_inside = slam.cfg['tracking']['f2m']['vis_inside']
        vis_dir = os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis')
        self.f2m_visualizer = Visualizer(
            freq=slam.cfg['tracking']['f2m']['vis_freq'],
            inside_freq=slam.cfg['tracking']['f2m']['vis_inside_freq'],
            vis_dir=vis_dir,
            renderer=self.renderer,
            verbose=self.verbose,
            device=self.device,
            wandb_logger=self.wandb_logger,
            vis_inside=self.vis_inside,
            total_iters=self.num_cam_iters,
            is_tracker=True)

        self.npc = slam.npc
        self.shared_decoders = slam.conv_o_net
        self.exposure_feat = slam.exposure_feat[0].clone().requires_grad_()

        self.decoders = None
        self.npc_geo_feats = None
        self.npc_col_feats = None
        self.cloud_pos = None

        self.prev_mapping_idx = -1
        self.mapping_idx = slam.mapping_idx

        self.cfg_only_kf = slam.cfg['tracking']['f2m']['only_kf']

        if self.use_dynamic_radius:
            os.makedirs(f'{self.output}/dynamic_r_frame_f2m', exist_ok=True)

    def track(self, idx, gt_color, gt_depth, gt_c2w, init_pose, is_kf=False, rgb_only=False, vis=False):
        """
        Performs the frame-to-model tracking and returns the result of the optimization as the pose prediction.

        Args:
            idx (int): frame index/timestamp
            gt_color (tensor): ground truth color image, shape (H, W, 3)
            gt_depth (tensor): ground truth depth image, shape (H, W)
            gt_c2w (tensor): ground truth camera, 4x4 matrix [R(3,3),T(3,1);0(1,3),1]. relative to first frame camera pose
            init_pose (tensor): pose to initialize for optimization. defaults to None. [tx,ty,tz,qx,qy,qz,qw].
            is_kf (bool): whether the frame is a keyframe. defaults to False.
            rgb_only (bool): whether to use only RGB information for tracking. defaults to False.
        """
        if self.cfg_only_kf and not is_kf:
            return None

        if self.use_dynamic_radius:
            ratio = self.radius_query_ratio
            intensity = rgb2gray(gt_color.cpu().numpy())
            grad_y = filters.sobel_h(intensity)
            grad_x = filters.sobel_v(intensity)
            color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            color_grad_mag = np.clip(
                color_grad_mag, 0.0, self.color_grad_threshold)
            fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                    self.radius_add_max, self.radius_add_max, self.radius_add_min])
            fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                        ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
            dynamic_r_add = fn_map_r_add(color_grad_mag)
            dynamic_r_query = fn_map_r_query(color_grad_mag)
            self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                self.device), torch.from_numpy(dynamic_r_query).to(self.device)
            torch.save(self.dynamic_r_query,
                        f'{self.output}/dynamic_r_frame_f2m/r_query_{idx:05d}.pt')
            
        if self.sample_with_color_grad:
            H, W = self.H, self.W
            Wedge = self.ignore_edge_W
            Hedge = self.ignore_edge_H
            selected_index, cur_color_grad = get_selected_index_with_grad(Hedge, H-Hedge, Wedge, W-Wedge,
                                                                            self.tracking_pixels, gt_color,
                                                                            gt_depth=gt_depth, depth_limit=self.depth_limit)
        else:
            selected_index = None

        if self.verbose:
            print(Fore.MAGENTA)
            print(">>> F2M-Tracking Frame ", idx)
            print(Style.RESET_ALL)

        gt_c2w = gt_c2w.clone()
        gt_c2w[:3, 1:3] *= -1
        gt_camera_tensor = get_tensor_from_camera(gt_c2w).to(self.device)  # [quad,T]

        if idx==0:
            logger.debug("pose set to Id")
            pose = lietorch.SE3.Identity(1)[0].data.to(self.device).float()
        else:
            estimated_new_c2w = lietorch.SE3(init_pose).matrix().reshape(4, 4).to(self.device)
            estimated_new_c2w[:3, 1:3] *= -1
            logger.debug(f'starting c2w for optimization:\n{estimated_new_c2w}')

            camera_tensor = get_tensor_from_camera(estimated_new_c2w.detach())  # [quad,T]
            if torch.dot(camera_tensor[:4], gt_camera_tensor[:4]).item() < 0:
                camera_tensor[:4] *= -1
            
            if idx<=1:
                # bigger step for the first frame
                num_cam_iters = self.num_cam_iters * 4
            else:
                num_cam_iters = self.num_cam_iters

            camera_tensor = Variable(camera_tensor.to(self.device), requires_grad=True)

            if self.separate_LR:
                camera_tensor = camera_tensor.to(self.device).detach()
                T = camera_tensor[-3:]
                quad = camera_tensor[:4]
                self.quad = Variable(quad, requires_grad=True)
                self.T = Variable(T, requires_grad=True)
                camera_tensor = torch.cat([quad, T], 0)
                optim_para_list = [{'params': [self.T], 'lr': self.cam_lr},
                                   {'params': [self.quad], 'lr': self.cam_lr*0.2}]
            else:
                camera_tensor = Variable(camera_tensor.to(self.device), requires_grad=True)
                optim_para_list = [{'params': [camera_tensor], 'lr': self.cam_lr}]

            if self.encode_exposure:
                optim_para_list.append(
                    {'params': self.exposure_feat, 'lr': 0.001})
                optim_para_list.append(
                    {'params': self.decoders.color_decoder.mlp_exposure.parameters(), 'lr': 0.001})
            optimizer_camera = torch.optim.Adam(optim_para_list)
            
            initial_loss_camera_tensor = torch.abs(gt_camera_tensor.to(self.device)-camera_tensor)  # for print/log
            candidate_cam_tensor = None
            current_min_loss = float(1e20)

            initial_camera_tensor = camera_tensor.detach().clone()  # only for print/log
            logger.debug(f"Initial tensor ([Qwxyz,Txyz]): {initial_camera_tensor.detach().cpu().numpy().round(4)}")

            logger.debug(f"looping f2m optimization {num_cam_iters=}")
            for cam_iter in range(num_cam_iters):
                if self.separate_LR:
                    camera_tensor = torch.cat(
                        [self.quad, self.T], 0).to(self.device)
                    
                loss, color_loss_pixel, geo_loss_pixel = self.optimize_cam_in_batch(
                    camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, selected_index=selected_index, rgb_only=rgb_only)
                
                loss_camera_tensor = torch.abs(gt_camera_tensor.to(self.device)-camera_tensor).mean().item()  # for print/log

                if cam_iter == 0:
                    initial_loss = loss

                if loss < current_min_loss or cam_iter == 0:
                    current_min_loss = loss
                    candidate_cam_tensor = camera_tensor.detach().clone()  # [quad,T]

                if cam_iter == num_cam_iters - 1:
                    idx_loss_camera_tensor = torch.abs(gt_camera_tensor.to(self.device)-candidate_cam_tensor)  # for print/log
                    print(f'idx:{idx}, re-rendering loss: {initial_loss:.2f}->{current_min_loss:.2f}, ' +
                                f'camera_quad_error: {initial_loss_camera_tensor[:4].mean().item():.4f}->{idx_loss_camera_tensor[:4].mean().item():.4f}, '
                                + f'camera_pos_error: {initial_loss_camera_tensor[-3:].mean().item():.4f}->{idx_loss_camera_tensor[-3:].mean().item():.4f}')
                    if self.wandb_logger:
                        self.wandb_logger.log({
                            #'camera_quad_error': idx_loss_camera_tensor[:4].mean().item(),  # logged in PoseErrorEvaler outside F2M block
                            #'camera_pos_error': idx_loss_camera_tensor[-3:].mean().item(),
                            'color_loss_tracker': color_loss_pixel,
                            'geo_loss_tracker': geo_loss_pixel,
                            })
                else:
                    if cam_iter % 20 == 0:
                        print(f'iter: {cam_iter}, camera tensor error: {loss_camera_tensor:.4f}')

                if self.vis_inside and vis:
                    self.f2m_visualizer.vis_trk(
                        idx, cam_iter, gt_depth,
                        gt_color, camera_tensor,
                        self.npc, self.decoders, self.npc_geo_feats, self.npc_col_feats,
                        freq_override=is_kf,  # force vis if KF
                        dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos,
                        exposure_feat=self.exposure_feat)

            logger.debug(f"Initial tensor: {initial_camera_tensor.detach().cpu().numpy().round(4)}")
            logger.debug(f"  Final tensor: {candidate_cam_tensor.detach().cpu().numpy().round(4)}")
            logger.debug(f"     GT tensor: {gt_camera_tensor.detach().cpu().numpy().round(4)}")
            
            bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4)
            c2w = get_camera_from_tensor(candidate_cam_tensor.detach().clone())  # [quad,T] -> 4x4 matrix
            c2w = torch.cat([c2w, bottom], dim=0)

            if (not self.vis_inside) and vis:
                self.f2m_visualizer.vis_trk(
                    idx, num_cam_iters-1, gt_depth,
                    gt_color, c2w,
                    self.npc, self.decoders, self.npc_geo_feats, self.npc_col_feats,
                    freq_override=is_kf,  # force vis if KF
                    dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos,
                    exposure_feat=self.exposure_feat, cur_total_iters=num_cam_iters)

            c2w = c2w.detach().clone()
            c2w[:3, 1:3] *= -1
            pose = get_tensor_from_camera(c2w, Tquad=True)  # [T,quad]

        if self.verbose:
            print(Fore.YELLOW)
            print("<<< F2M-Tracked Frame ", idx)
            print(Style.RESET_ALL)

        return pose.clone().detach().to(self.device)  # (7,) pose
    
    
    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            self.decoders = self.shared_decoders
            self.npc_geo_feats = self.npc.get_geo_feats().detach().clone()
            self.npc_col_feats = self.npc.get_col_feats().detach().clone()
            self.prev_mapping_idx = self.mapping_idx[0].clone()
            self.cloud_pos = torch.tensor(
                self.npc.cloud_pos(), device=self.device).reshape(-1, 3)
            if self.verbose:
                print('Tracker has updated the parameters from Mapper.')


    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer,
                              selected_index=None, rgb_only=False):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor. [quad,T].
            gt_color (tensor): ground truth color image of the current frame, shape (H, W, 3)
            gt_depth (tensor): ground truth depth image of the current frame, shape (H, W)
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.
            selected_index: top color gradients pixels are pre-selected.
            rgb_only (bool): whether to use only RGB information for tracking.

        Returns:
            loss (float): total loss
            color_loss (float): color loss component
            geo_loss (float): geometric loss component
        """
        assert not rgb_only or self.use_color_in_tracking, 'RGB-only tracking requires color information.'

        device = self.device
        npc = self.npc
        H, W = self.H, self.W
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        if self.sample_with_color_grad:
            sample_size = batch_size
            cur_samples = np.random.choice(
                range(0, selected_index.shape[0]), size=sample_size, replace=False)

            index_color_grad = selected_index[cur_samples]
            i, j = np.unravel_index(index_color_grad.astype(int), (H, W))
            i, j = torch.from_numpy(j).to(device).float(
            ), torch.from_numpy(i).to(device).float()
            batch_rays_o, batch_rays_d = get_rays_from_uv(
                i, j, c2w, self.fx, self.fy, self.cx, self.cy, device)
            i, j = i.long(), j.long()
            batch_gt_depth = gt_depth[j, i]
            batch_gt_color = gt_color[j, i]

        else:
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                Hedge, H - Hedge, Wedge, W - Wedge,
                batch_size,
                self.fx, self.fy, self.cx, self.cy, c2w, gt_depth, gt_color, device,
                depth_filter=True, return_index=True, depth_limit=5.0 if self.depth_limit else None)

        if self.use_dynamic_radius:
            batch_r_query = self.dynamic_r_query[j, i]
        assert torch.numel(
            batch_gt_depth) != 0, 'gt_depth after filter is empty, please check.'
        
        with torch.no_grad():
            inside_mask = batch_gt_depth <= torch.minimum(
                10*batch_gt_depth.median(), 1.2*torch.max(batch_gt_depth))
        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]
        batch_r_query = batch_r_query[inside_mask] if self.use_dynamic_radius else None

        ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o,
                                             device, stage='color',  gt_depth=batch_gt_depth,
                                             npc_geo_feats=self.npc_geo_feats,
                                             npc_col_feats=self.npc_col_feats,
                                             is_tracker=True, cloud_pos=self.cloud_pos,
                                             dynamic_r_query=batch_r_query,
                                             exposure_feat=self.exposure_feat)
        depth, uncertainty, color, valid_ray_mask, _ = ret

        uncertainty = uncertainty.detach()
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        # remove pixels seen as outlier
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.mean()) & (batch_gt_depth > 0)
        else:
            tmp = torch.abs(batch_gt_depth-depth)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)

        mask = mask & nan_mask

        uncertainty_regul = torch.log(torch.sqrt(uncertainty)+1e-10)
        geo_loss_pixel = torch.abs(batch_gt_depth-depth) / torch.sqrt(uncertainty+1e-10) + uncertainty_regul
        geo_loss = torch.clamp(geo_loss_pixel, min=0.0, max=1e3)[mask].sum()

        color_loss_pixel = torch.abs(batch_gt_color - color)
        color_loss = color_loss_pixel[mask].sum()

        if rgb_only:
            loss = color_loss
        else:
            loss = geo_loss
            if self.use_color_in_tracking:
                loss += self.w_color_loss*color_loss

        #logger.debug(f'geo_loss={geo_loss.item()} color_loss={color_loss.item()} loss={loss.item()}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), (color_loss/mask.shape[0]).item(), (geo_loss/mask.shape[0]).item()
