import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor, get_tensor_from_camera
import wandb
import cv2
from src.neural_point import proj_depth_map


class Visualizer(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0', wandb_logger=None,
                 vis_inside=False, total_iters=None, img_dir=None, is_tracker=False):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.wandb_logger = wandb_logger
        self.vis_inside = vis_inside
        self.total_iters = total_iters
        self.img_dir = img_dir
        self.is_tracker = is_tracker
        os.makedirs(f'{vis_dir}', exist_ok=True)

    @torch.no_grad()
    def vis_value_only(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, npc,
                       decoders, npc_geo_feats, npc_col_feats, freq_override=False,
                       dynamic_r_query=None, cloud_pos=None, exposure_feat=None,
                       use_coarse_pcl=False):
        """
        return rendered depth and color map only
        """
        if freq_override or (idx % self.freq == 0):
            if len(c2w_or_camera_tensor.shape) == 1:
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                    torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    c2w_or_camera_tensor.detach().clone())
                c2w = torch.cat([c2w, bottom], dim=0)
            else:
                c2w = c2w_or_camera_tensor

            depth, uncertainty, color, valid_ray_mask, valid_ray_count = self.renderer.render_img(
                npc,
                decoders,
                c2w,
                self.device,
                stage='color',
                gt_depth=gt_depth, npc_geo_feats=npc_geo_feats,
                npc_col_feats=npc_col_feats,
                dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos,
                exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl,
                is_tracker=self.is_tracker)
            return depth, color, valid_ray_mask, valid_ray_count

    @torch.no_grad()
    def vis(self, idx, iter, gt_depth,
            render_depth, droid_depth, mono_depth,
            gt_color, c2w_or_camera_tensor, npc,
            decoders, npc_geo_feats, npc_col_feats, cfg,
            freq_override=False,
            dynamic_r_query=None, cloud_pos=None, exposure_feat=None,
            cur_total_iters=None, save_rendered_image=False):
        """
        Visualization of all depth, color and PC images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            render_depth (): TODO
            droid_depth (): TODO
            mono_depth (): TODO
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            npc (): neural point cloud.
            decoders (nn.module): decoders.
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            cfg (): TODO
            freq_override (bool): call vis() at will
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.
            cur_total_iters (int): number of iterations done when saving
            save_rendered_image (bool): whether to save the rgb image in separate folder apart from the standard visualization
        """
        total_iters = cur_total_iters if cur_total_iters is not None else self.total_iters
        if self.vis_inside:
            conditions = (idx > 0 and (
                (iter % self.inside_freq == 0) or (iter == total_iters-1))) or freq_override
        else:
            conditions = (idx > 0 and (
                iter == total_iters-1)) or freq_override
        if conditions:
            gt_depth_np = gt_depth.cpu().numpy()
            render_depth_np = render_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            if len(c2w_or_camera_tensor.shape) == 1:
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                    torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    c2w_or_camera_tensor.detach().clone())
                c2w = torch.cat([c2w, bottom], dim=0)
            else:
                c2w = c2w_or_camera_tensor

            depth, uncertainty, color, valid_ray_mask, valid_ray_count = self.renderer.render_img(
                npc,
                decoders,
                c2w,
                self.device,
                stage='color',
                gt_depth=render_depth, npc_geo_feats=npc_geo_feats,
                npc_col_feats=npc_col_feats,
                dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos,
                exposure_feat=exposure_feat,
                is_tracker=self.is_tracker)
            
            if save_rendered_image and self.img_dir is not None:
                img = cv2.cvtColor(color.cpu().numpy()
                                    * 255, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(
                    f'{self.img_dir}', f'frame_{idx:05d}.png'), img)
                # img_mask = cv2.cvtColor(valid_ray_mask.cpu().float().numpy()
                #                    * 255, cv2.COLOR_GRAY2BGR)
                # cv2.imwrite(os.path.join(
                #     f'{self.img_dir}', f'mask_{idx:05d}.png'), img_mask)

            depth_np = depth.detach().cpu().numpy()
            color = torch.round(color*255.0)/255.0
            color_np = color.detach().cpu().numpy()
            depth_residual = np.abs(render_depth_np - depth_np)
            depth_residual[render_depth_np == 0.0] = 0.0
            depth_residual = np.clip(depth_residual, 0.0, 0.2)

            color_residual = np.abs(gt_color_np - color_np)
            color_residual[render_depth_np == 0.0] = 0.0
            droid_depth_np = droid_depth.detach().cpu().numpy()
            proj_depth = proj_depth_map(c2w.clone(),npc,self.device,cfg)
            proj_depth_np = proj_depth.detach().cpu().numpy()
            fig, axs = plt.subplots(5, 3)
            # fig.tight_layout()
            max_depth = np.max(droid_depth_np)
            # max_depth = 10.0
            axs[0, 0].imshow(render_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Rendered Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 2].imshow(depth_residual, cmap="plasma")
            axs[0, 2].set_title('Depth Residual')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)
            color_residual = np.clip(color_residual, 0, 1)
            axs[1, 0].imshow(gt_color_np, cmap="plasma")
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(color_np, cmap="plasma")
            axs[1, 1].set_title('Rendered RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

            valid_ray_count_np = valid_ray_count.detach().cpu().numpy()
            proj_depth_mask = proj_depth > 0
            proj_depth_mask_np = proj_depth_mask.detach().cpu().int().numpy()

            axs[2, 0].imshow(droid_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[2, 0].set_title('Droid depth')
            axs[2, 0].set_xticks([])
            axs[2, 0].set_yticks([])
            axs[2, 1].imshow(proj_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[2, 1].set_title('Pointcloud depth')
            axs[2, 1].set_xticks([])
            axs[2, 1].set_yticks([])
            axs[2, 2].imshow(proj_depth_mask_np, cmap="plasma",
                                vmin=0, vmax=1)
            axs[2, 2].set_title('Pointcloud mask')
            axs[2, 2].set_xticks([])
            axs[2, 2].set_yticks([])

            axs[3, 0].imshow(valid_ray_count_np, cmap="plasma",
                                vmin=0, vmax=cfg["rendering"]["N_surface_mapper"])
            axs[3, 0].set_title('valid counts')
            axs[3, 0].set_xticks([])
            axs[3, 0].set_yticks([])

            s_proj_depth = proj_depth_map(c2w,npc,self.device,cfg, neural_pcl=True)
            s_proj_depth_np = s_proj_depth.detach().cpu().numpy()
            axs[3, 1].imshow(s_proj_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[3, 1].set_title('Sparse Pointcloud depth')
            axs[3, 1].set_xticks([])
            axs[3, 1].set_yticks([])

            axs[3, 2].axis('off')

            axs[4, 0].imshow(gt_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[4, 0].set_title('GT depth')
            axs[4, 0].set_xticks([])
            axs[4, 0].set_yticks([])

            mono_depth_np = mono_depth.detach().cpu().numpy()
            axs[4, 1].imshow(mono_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[4, 1].set_title('Monocular depth')
            axs[4, 1].set_xticks([])
            axs[4, 1].set_yticks([])

            axs[4, 2].axis('off')


            plt.subplots_adjust(wspace=0, hspace=0.5)
            fig_name = f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg'
            plt.savefig(fig_name, dpi=300,
                        bbox_inches='tight', pad_inches=0.1)
            if 'mapping' in self.vis_dir and self.wandb_logger:
                self.wandb_logger.log(
                    ({f'Mapping_{idx:05d}_{iter:04d}': wandb.Image(fig_name)}))
            plt.clf()
            plt.close()

            if self.verbose:
                print(f'Saved full rendering visualization of color/depth image at {fig_name}')

    @torch.no_grad()
    def vis_trk(self, idx, iter, gt_depth,
                    gt_color, c2w_or_camera_tensor, npc,
                    decoders, npc_geo_feats, npc_col_feats,
                    freq_override=False,
                    dynamic_r_query=None, cloud_pos=None, exposure_feat=None,
                    cur_total_iters=None, save_rendered_image=False):
        """
        Visualization of depth, color images and save to file.
        For tracking.
        Only rendered-RGBD and GT-RGBD.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor ([quad,T]).
            npc (): neural point cloud.
            decoders (nn.module): decoders.
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            freq_override (bool): call vis() at will
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.
            cur_total_iters (int): number of iterations done when saving
            save_rendered_image (bool): whether to save the rgb image in separate folder apart from the standard visualization
        """
        total_iters = cur_total_iters if cur_total_iters is not None else self.total_iters
        if self.vis_inside:
            conditions = (idx > 0 and (idx % self.freq == 0) and (
                (iter % self.inside_freq == 0) or (iter == total_iters-1))) or freq_override
        else:
            conditions = (idx > 0 and (idx % self.freq == 0) and (
                iter == total_iters-1)) or freq_override
        if conditions:
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            if len(c2w_or_camera_tensor.shape) == 1:
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                    torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    c2w_or_camera_tensor.detach().clone())
                c2w = torch.cat([c2w, bottom], dim=0)
            else:
                c2w = c2w_or_camera_tensor

            depth, uncertainty, color, valid_ray_mask, valid_ray_count = self.renderer.render_img(
                npc,
                decoders,
                c2w,
                self.device,
                stage='color',
                gt_depth=gt_depth, npc_geo_feats=npc_geo_feats,
                npc_col_feats=npc_col_feats,
                dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos,
                exposure_feat=exposure_feat,
                is_tracker=self.is_tracker)
            if save_rendered_image and self.img_dir is not None:
                img = cv2.cvtColor(color.cpu().numpy()
                                    * 255, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(
                    f'{self.img_dir}', f'frame_{idx:05d}.png'), img)
                img_mask = cv2.cvtColor(valid_ray_mask.cpu().float().numpy()
                                   * 255, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(
                    f'{self.img_dir}', f'mask_{idx:05d}.png'), img_mask)

            depth_np = depth.detach().cpu().numpy()
            color = torch.round(color*255.0)/255.0
            color_np = color.detach().cpu().numpy()
            valid_count_np = valid_ray_count.detach().cpu().numpy()
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np == 0.0] = 0.0
            depth_residual = np.clip(depth_residual, 0.0, 0.05)

            color_residual = np.abs(gt_color_np - color_np)
            color_residual[gt_depth_np == 0.0] = 0.0

            fig, axs = plt.subplots(3, 3)
            fig.tight_layout()
            max_depth = np.max(gt_depth_np)
            axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(depth_np, cmap="plasma",
                                vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Rendered Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 2].imshow(depth_residual, cmap="plasma")
            axs[0, 2].set_title('Depth Residual')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)
            color_residual = np.clip(color_residual, 0, 1)
            axs[1, 0].imshow(gt_color_np, cmap="plasma")
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(color_np, cmap="plasma")
            axs[1, 1].set_title('Rendered RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
            axs[2, 0].imshow(valid_count_np, cmap="plasma")
            axs[2, 0].set_title('Valid ray count')
            axs[2, 0].set_xticks([])
            axs[2, 0].set_yticks([])
            axs[2, 1].axis('off')
            axs[2, 2].axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            fig_name = f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg'
            plt.savefig(fig_name, dpi=300,
                        bbox_inches='tight', pad_inches=0.1)
            if 'tracking' in self.vis_dir and self.wandb_logger:
                self.wandb_logger.log(
                    ({f'Tracking_{idx:05d}_{iter:04d}': wandb.Image(fig_name)}))
            plt.clf()
            plt.close()

            # #>>>DEBUG POSE
            # cam_tensor = get_tensor_from_camera(c2w, Tquad=True)
            # cam_tensor = cam_tensor.cpu().numpy().reshape(-1).round(4).tolist()
            # dir_ = self.vis_dir.rstrip('/').rsplit('/', 1)[0]
            # pose_filename = f'{dir_}/pose_debug.txt'
            # with open(pose_filename, 'a') as f:
            #     mode = 'trk' if 'tracking' in self.vis_dir else 'map'
            #     f.write(f'{idx:05d}_{iter:04d}-{mode} {cam_tensor}\n')
            #     f.write(f'\tinput {gt_depth_np.min():.4f}-{gt_depth_np.max():.4f}\n')
            #     f.write(f'\trendr {depth_np.min():.4f}-{depth_np.max():.4f}\n')
            # #<<<

            if self.verbose:
                print(
                    f'Saved minimal rendering visualization of color/depth image at {fig_name}')


import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=0.5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def add_traj(self,traj):
        traj_length = len(traj)
        xs = np.array([traj[i][0,3] for i in range(traj_length)])
        ys = np.array([traj[i][1,3] for i in range(traj_length)])
        zs = np.array([traj[i][2,3] for i in range(traj_length)])

        x_length = xs.max()-xs.min()
        y_length = ys.max()-ys.min()
        z_length = zs.max()-zs.min()

        xlim = [xs.min()-0.*x_length, xs.max()+0.*x_length]
        ylim = [ys.min()-0.*y_length, ys.max()+0.*y_length]
        zlim = [zs.min()-0.*z_length, zs.max()+0.*z_length]
        
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        for i in range(traj_length):
            self.extrinsic2pyramid(traj[i],plt.cm.rainbow(i / traj_length), 0.3)
        # self.colorbar(traj_length)


    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')
    
    def save(self,path):
        self.fig.savefig(path)
        print(f"Camera 3D trajectory is saved: {path}")
        plt.clf()
    