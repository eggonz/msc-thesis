import os

import torch
import lietorch
import cv2

from src.utils.eval_render import get_projected_depth
from src.utils.Renderer import Renderer

    
class PsnrEvaler:
    def __init__(self, slam, device, full_res=False):  # NOTE: change with psnr-resolution
        self.cfg = slam.cfg
        self.npc = slam.npc
        self.decoders = slam.conv_o_net
        self.device = device
        self.renderer = Renderer(self.cfg, full_res=full_res)
        self.output = slam.output
        self.wandb_logger = slam.wandb_logger
        self.vis_dir = os.path.join(self.output, 'psnrx_vis')
        os.makedirs(self.vis_dir, exist_ok=True)

        self._full_res = full_res

    
    def __call__(self, gt_color, gt_depth, c2w_pose, save_name=None):
        """
        Args:
            gt_color: torch.Tensor, shape=(H, W, 3), input color image, matches chosen reoslution
            gt_depth: torch.Tensor, shape=(H, W), input (mono) depth, matches chosen reoslution
            c2w_pose: torch.Tensor, shape=(7,) c2w pose tensor
            save_name: str, save path suffix. If None, no save. '{self.vis_dir}/frame_{save_name}.png'
        """
        est_c2w = lietorch.SE3(c2w_pose).matrix().detach().cpu().numpy()

        # project npc, fill with gt_depth
        rendered_depth, cur_c2w, r_query_frame = get_projected_depth(
            gt_color, est_c2w, gt_depth, self.npc, self.device, self.cfg, full_res=self._full_res)

        # render npc
        depth, uncertainty, color, valid_ray_mask, valid_ray_count = self.renderer.render_img(
            self.npc,
            self.decoders,  # decoders
            cur_c2w,  # 4x4
            self.device,
            stage='color',
            gt_depth=gt_depth,  # reference for fustrum limits
            npc_geo_feats=self.npc.get_geo_feats(),
            npc_col_feats=self.npc.get_col_feats(),
            dynamic_r_query=r_query_frame,
            cloud_pos=self.npc.cloud_pos(),
            exposure_feat=None,  # cfg.model.encode_exposure=False
            is_tracker=True,  # use mapper visualization config
        )

        mask = (valid_ray_mask>0) * (rendered_depth > 0) * (gt_depth>0) # * (droid_depth>0)

        if mask.sum() == 0:
            print('No valid pixels found in mask')
            return {
                'psnr': -1,
                'psnr_masked': -1,
            }

        if save_name is not None and self.vis_dir is not None:
            img = cv2.cvtColor(color.cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(f'{self.vis_dir}', f'{save_name}_frame.png'), img)
            img_mask = cv2.cvtColor(valid_ray_mask.cpu().float().numpy() * 255, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(f'{self.vis_dir}', f'{save_name}_mask.png'), img_mask)
            residual = torch.abs(gt_color - color)
            residual[~mask] = 0
            residual = torch.clamp(residual, 0, 1)
            res = cv2.cvtColor(residual.cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(f'{self.vis_dir}', f'{save_name}_residual.png'), res)

        mse_loss = torch.nn.functional.mse_loss(gt_color, color)
        psnr_frame = -10. * torch.log10(mse_loss)

        masked_mse_loss = torch.nn.functional.mse_loss(gt_color[mask], color[mask])
        masked_psnr_frame = -10. * torch.log10(masked_mse_loss)

        # TODO add more metrics
        metrics = {
            'psnr': psnr_frame.item(),
            'psnr_masked': masked_psnr_frame.item(),
        }
        return metrics
