import os
import abc
import shutil
import torch
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.utils.datasets import load_mono_depth
from src.neural_point import proj_depth_map, get_droid_render_depth
from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim
import traceback
import numpy as np
from src.utils.pose_trajectory import OutputTrajectory

def mono_scale_and_shift(mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
    # mono_depth: [B, n]
    # weights: [B, n]
    if weights is None:
        weights = torch.ones_like(mono_depth).to(mono_depth.device)
    n = mono_depth.shape[-1]
    mono_depth = mono_depth.reshape(-1,n)
    est_depth = est_depth.reshape(-1,n)
    weights = weights.reshape(-1,n)

    prediction = mono_depth
    target = est_depth  

    a_00 = torch.sum(weights * prediction * prediction, dim=[1])
    a_01 = torch.sum(weights * prediction, dim=[1])
    a_11 = torch.sum(weights, dim=[1])

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1])
    b_1 = torch.sum(weights * target, dim=[1])

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01

    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det

    return scale, shift


def get_projected_depth(gt_color, est_c2w, mono_depth, npc, device, cfg, full_res=False):
        """
        Args:
            image: input color image, gt_color [h,w,3]
            est_c2w: numpy 4x4 c2w matrix, e.g. est_c2w = lietorch.SE3(self.video.pose_w2c).inv().matrix().numpy()
            mono_depth: [h,w]
            npc # mapper.npc
            device # mapper.device
            cfg # mapper.cfg
        """

        use_dynamic_radius = cfg['pointcloud']['use_dynamic_radius']
        radius_add_max = cfg['pointcloud']['radius_add_max']
        radius_add_min = cfg['pointcloud']['radius_add_min']
        radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
        color_grad_threshold = cfg['pointcloud']['color_grad_threshold']

        cur_c2w = torch.from_numpy(est_c2w.copy()).to(device)  # c2w is relative
        cur_c2w[:3, 1:3] *= -1
        proj_depth = proj_depth_map(cur_c2w, npc, device, cfg, full_res=full_res)
        proj_valid_mask = (proj_depth > 0)

        render_depth = proj_depth

        if proj_valid_mask.sum() > 0:
            mono_scale, mono_shift = mono_scale_and_shift(mono_depth[proj_valid_mask],proj_depth[proj_valid_mask],None)
            mono_depth_wq = mono_scale*mono_depth + mono_shift
            render_depth[~proj_valid_mask] = mono_depth_wq[~proj_valid_mask]

        r_query_frame = None

        if use_dynamic_radius:
            ratio = radius_query_ratio
            intensity = rgb2gray(gt_color.cpu().numpy())
            grad_y = filters.sobel_h(intensity)
            grad_x = filters.sobel_v(intensity)
            color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
            color_grad_mag = np.clip(
                color_grad_mag, 0.0, color_grad_threshold)  # range 0~1
            fn_map_r_query = interp1d([0, 0.01, color_grad_threshold], [
                                    ratio*radius_add_max, ratio*radius_add_max, ratio*radius_add_min])
            dynamic_r_query = fn_map_r_query(color_grad_mag)

            r_query_frame = torch.from_numpy(dynamic_r_query).to(device)
            r_query_frame = r_query_frame/3.0 * render_depth

        return render_depth, cur_c2w, r_query_frame


class Evaler(abc.ABC):
    def __init__(self, mapper):
        # TODO call in tracker (online) instead of slam.terminate
        self.mapper = mapper

        self.frame_count = 0
        self.psnr_sum = 0
        self.ssim_sum = 0
        self.lpips_sum = 0
        self.masked_psnr_sum = 0
        self.masked_ssim_sum = 0
        self.masked_lpips_sum = 0

        if os.path.exists(self.render_dir):
            shutil.rmtree(self.render_dir)
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.rerender_dir, exist_ok=True)

        self.cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(mapper.device)

    @abc.abstractmethod
    def _get_render_depth(self, *args, **kwargs):
        raise NotImplementedError

    def process_frame(self, idx_frame, idx_kf=None, est_c2w=None):
        # timestamp, image, gt_depth, intrinsics, gt_c2w, gt_color_full, gt_depth_full = self.mapper.frame_reader[idx_frame]
        _, image, gt_depth, _, _, gt_color_full, gt_depth_full = self.mapper.frame_reader[idx_frame]

        # mono_depth = load_mono_depth(idx_frame,self.mapper.cfg)
        mono_depth = gt_depth.clone()
        gt_color = image.to(self.mapper.device).squeeze(0).permute(1,2,0)
        gt_depth = gt_depth.to(self.mapper.device)
        # gt_color = gt_color_full.to(self.mapper.device)
        # gt_depth = gt_depth_full.to(self.mapper.device)
        mono_depth = mono_depth.to(self.mapper.device)

        # Render depth
        if idx_kf is not None:
            # is keyframe, use keyframe depth
            render_depth, cur_c2w, r_query_frame = self._get_render_depth(idx_frame, idx_kf, mono_depth)
        else:
            # is arbitrary frame, use estimated pose
            render_depth, cur_c2w, r_query_frame = self._get_render_depth(gt_color, est_c2w, mono_depth)

        cur_frame_depth, cur_frame_color, valid_mask, valid_ray_count = self.mapper.visualizer.vis_value_only(idx_frame, 0, render_depth, gt_color, cur_c2w, self.mapper.npc, self.mapper.decoders,
                                                                            self.mapper.npc.get_geo_feats(), self.mapper.npc.get_col_feats(), freq_override=True,
                                                                            dynamic_r_query=r_query_frame, cloud_pos=self.mapper.cloud_pos_tensor,
                                                                            exposure_feat=self.mapper.exposure_feat_all[
                                                                                idx_frame // self.mapper.cfg["mapping"]["every_frame"]
                                                                            ].to(self.mapper.device)
                                                                            if self.mapper.encode_exposure else None)
        
        img = cv2.cvtColor(cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(os.path.join(f'{self.rerender_dir}', f'frame_{idx_frame:05d}.png'), img)

        mse_loss = torch.nn.functional.mse_loss(
            gt_color, cur_frame_color)
        psnr_frame = -10. * torch.log10(mse_loss)
        ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                data_range=1.0, size_average=True)
        lpips_value = self.cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0))
        self.psnr_sum += psnr_frame
        self.ssim_sum += ssim_value
        self.lpips_sum += lpips_value

        mask = (valid_mask>0) * (render_depth > 0) * (gt_depth>0) # * (droid_depth>0)
        cur_frame_depth[~mask] = 0.0 
        gt_color[~mask] = 0.0
        cur_frame_color[~mask] = 0.0

        np.save(f'{self.render_dir}/depth_{idx_frame:05d}', cur_frame_depth.cpu().numpy())
        np.save(f'{self.render_dir}/color_{idx_frame:05d}', cur_frame_color.cpu().numpy())

        masked_mse_loss = torch.nn.functional.mse_loss(gt_color[mask], cur_frame_color[mask])
        masked_psnr_frame = -10. * torch.log10(masked_mse_loss)
        masked_ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
                                data_range=1.0, size_average=True)
        masked_lpips_value = self.cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0))
        self.masked_psnr_sum += masked_psnr_frame
        self.masked_ssim_sum += masked_ssim_value
        self.masked_lpips_sum += masked_lpips_value

        self.frame_count += 1
        if self.frame_count % 20 == 0:
            print(f'frame {idx_frame}')
            
        return {
            "psnr": psnr_frame.item(),
            "ssim": ssim_value.item(),
            "lpips": lpips_value.item(),
            "masked_psnr": masked_psnr_frame.item(),
            "masked_ssim": masked_ssim_value.item(),
            "masked_lpips": masked_lpips_value.item(),
        }
    
    def compute_avg(self):
        avg_psnr = self.psnr_sum / self.frame_count
        avg_ssim = self.ssim_sum / self.frame_count
        avg_lpips = self.lpips_sum / self.frame_count
        avg_masked_psnr = self.masked_psnr_sum / self.frame_count
        avg_masked_ssim = self.masked_ssim_sum / self.frame_count
        avg_masked_lpips = self.masked_lpips_sum / self.frame_count
        return {
            "avg_psnr": avg_psnr.item(),
            "avg_ssim": avg_ssim.item(),
            "avg_lpips": avg_lpips.item(),
            "avg_masked_psnr": avg_masked_psnr.item(),
            "avg_masked_ssim": avg_masked_ssim.item(),
            "avg_masked_lpips": avg_masked_lpips.item(),
        }, self.frame_count


class EvalerKf(Evaler):
    def __init__(self, mapper):
        self.render_dir = f'{mapper.output}/rendered_every_keyframe'
        self.rerender_dir = f'{mapper.output}/rerendered_keyframe_image'
        super().__init__(mapper)

    def process_frame(self, idx_frame, idx_kf):
        return super().process_frame(idx_frame, idx_kf=idx_kf)

    def _get_render_depth(self, idx_frame, idx_kf, mono_depth):
        cur_c2w, mono_depth_wq,droid_depth = self.mapper.get_c2w_and_depth(idx_kf, idx_frame, mono_depth)  # c2w is relative
        if self.mapper.cfg["mapping"]["render_depth"] == "droid":
            render_depth = get_droid_render_depth(self.mapper.npc, self.mapper.cfg, cur_c2w.clone(), 
                                                    droid_depth, mono_depth_wq, self.mapper.device,
                                                    use_mono_to_complete=self.mapper.use_mono_to_complete)
        elif self.mapper.cfg["mapping"]["render_depth"] == "mono":
            render_depth = mono_depth_wq
        if self.mapper.encode_exposure:
            try:
                state_dict = torch.load(f'{self.mapper.output}/ckpts/color_decoder/{idx_frame:05}.pt',
                                        map_location=self.mapper.device)
                self.mapper.decoders.color_decoder.load_state_dict(
                    state_dict)
            except:
                print(
                    'Color decoder not loaded, will use saved weights in checkpoint.')

        r_query_frame = torch.load(f'{self.mapper.output}/dynamic_r_frame/r_query_{idx_frame:05d}.pt', map_location=self.mapper.device) \
            if self.mapper.use_dynamic_radius else None
        if r_query_frame is not None:
            r_query_frame = r_query_frame/3.0 * render_depth
        # mean_r_query = torch.mean(self.mapper.dynamic_r_query)
        # r_query_frame[r_query_frame==0] = mean_r_query

        return render_depth, cur_c2w, r_query_frame


class EvalerEvery(Evaler):
    def __init__(self, mapper):
        self.render_dir = f'{mapper.output}/rendered_every_frame'
        self.rerender_dir = f'{mapper.output}/rerendered_image'
        super().__init__(mapper)

    def process_frame(self, idx_frame, est_c2w):
        return super().process_frame(idx_frame, est_c2w=est_c2w)

    def _get_render_depth(self, gt_color, est_c2w, mono_depth):
        return get_projected_depth(gt_color, est_c2w, mono_depth, self.mapper.npc, self.mapper.device, self.mapper.cfg)


def eval_kf_imgs(self,scene_scale):
    # re-render frames at the end for meshing
    print('Starting re-rendering keyframes...')

    evaler = EvalerKf(self)
    
    try:
        for kf in self.keyframe_dict:

            metrics = evaler.process_frame(kf['idx'], kf['video_idx'])
            # TODO log these metrics. locally??

            # # NOTE: PSNR is logged online
            # if self.wandb_logger:
            #     self.wandb_logger.log({'psnr_frame': metrics['masked_psnr']})

        avg_metrics, count = evaler.compute_avg()

        output_str = '\n'.join([f"{key}: {value}" for key, value in avg_metrics.items()])
        print('###############\n' + output_str + '\n###############')

        if self.wandb_logger:
            self.wandb_logger.log(avg_metrics)

        out_path=f'{self.output}/logs/metrics_render_kf.txt'
        with open(out_path, 'w+') as fp:
            fp.write(output_str + '\n###############\n')
            
        print(f'Finished rendering {count} frames.')

    except Exception as e:
        traceback.print_exception(e)
        print('Rerendering frames failed.')


def eval_imgs(self,scene_scale,est_traj: OutputTrajectory):  # FIXME incomplete trajectories
    # re-render frames at the end for meshing
    print('Starting re-rendering frames...')

    evaler = EvalerEvery(self)
    est_timestamps, est_c2ws = est_traj.get_trajectory()
    
    try:
        for t, c2w in sorted(zip(est_timestamps, est_c2ws), key=lambda x: x[0]):
            idx = int(t)
            if idx % self.every_frame != 0:
                continue
            metrics = evaler.process_frame(idx, est_c2w=c2w)

        avg_metrics, count = evaler.compute_avg()
        avg_metrics_every = {f'{key}_every': value for key, value in avg_metrics.items()}

        output_str = '\n'.join([f"{key}: {value}" for key, value in avg_metrics.items()])
        output_str_every = '\n'.join([f"{key}: {value}" for key, value in avg_metrics_every.items()])
        print('###############\n' + output_str_every + '\n###############\n')

        if self.wandb_logger:
            self.wandb_logger.log(avg_metrics_every)

        out_path=f'{self.output}/logs/metrics_render_every.txt'
        with open(out_path, 'w+') as fp:
            fp.write(output_str + '\n###############\n')

        print(f'Finished rendering {count} frames.')

    except Exception as e:
        traceback.print_exception(e)
        print('Rerendering frames failed.')
