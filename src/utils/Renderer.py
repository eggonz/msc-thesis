import torch
import warnings
from src.common import get_rays, raw2outputs_nerf_color, update_cam


class Renderer(object):
    def __init__(self, cfg, points_batch_size=500000, ray_batch_size=3000, full_res=False):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.N_surface_mapper = cfg['rendering']['N_surface_mapper']
        self.near_end_surface_mapper = cfg['rendering']['near_end_surface_mapper']
        self.far_end_surface_mapper = cfg['rendering']['far_end_surface_mapper']
        self.N_surface_tracker = cfg['rendering']['N_surface_tracker']
        self.near_end_surface_tracker = cfg['rendering']['near_end_surface_tracker']
        self.far_end_surface_tracker = cfg['rendering']['far_end_surface_tracker']
        self.sample_near_pcl = cfg['rendering']['sample_near_pcl']
        self.sigmoid_coefficient:float = cfg['rendering']['sigmoid_coef_mapper']
        self.near_end = cfg['rendering']['near_end']

        self.use_dynamic_radius = cfg['pointcloud']['use_dynamic_radius']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg, full_res)

    def eval_points(self, p, decoders, npc, stage='color', device=None,
                    npc_geo_feats=None, npc_col_feats=None,
                    is_tracker=False, cloud_pos=None,
                    pts_views_d=None, ray_pts_num=None,
                    dynamic_r_query=None, exposure_feat=None,
                    use_coarse_pcl=False):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            npc (object): Neural point cloud.
            stage (str, optional): 'geometry'|'color', defaults to 'color'.
            device (str, optional): CUDA device.
            npc_geo_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            npc_col_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            is_tracker (bool, optional): tracker has different gradient flow in eval_points.
            cloud_pos (tensor, optional): positions of all point cloud features, used only when tracker calls.
            pts_views_d (tensor, optional): ray direction for each point
            ray_pts_num (tensor, optional): number of surface samples
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.

        Returns:
            ret (tensor): occupancy (and color) value of input points, (N,)
            valid_ray_mask (tensor): 
        """
        assert torch.is_tensor(p)
        if device == None:
            device = npc.device()
        p_split = torch.split(p, self.points_batch_size)
        rets = []
        ray_masks = []
        point_masks = []
        ray_counters = []
        for pi in p_split:
            pi = pi.unsqueeze(0)
            ret, valid_ray_mask, point_mask, valid_ray_counter = decoders(pi, npc, stage, npc_geo_feats, npc_col_feats,
                                                       ray_pts_num, is_tracker, cloud_pos, pts_views_d,
                                                       dynamic_r_query, exposure_feat, use_coarse_pcl)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            rets.append(ret)
            ray_masks.append(valid_ray_mask)
            point_masks.append(point_mask)
            ray_counters.append(valid_ray_counter)

        ret = torch.cat(rets, dim=0)
        ray_mask = torch.cat(ray_masks, dim=0)
        point_mask = torch.cat(point_masks, dim=0)
        ray_counter = torch.cat(ray_counters)

        return ret, ray_mask, point_mask, ray_counter

    def render_batch_ray(self, npc, decoders, rays_d, rays_o, device, stage, gt_depth=None,
                         npc_geo_feats=None, npc_col_feats=None, is_tracker=False, cloud_pos=None,
                         dynamic_r_query=None, exposure_feat=None, use_coarse_pcl=False):
        """
        Render color, depth, uncertainty and a valid ray mask from a batch of rays.

        Args:
            npc (): Neural point cloud.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None. input (N, )
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            is_tracker (bool, optional): tracker has different gradient flow in eval_points.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty (can be interpreted as epistemic uncertainty)
            color (tensor): rendered color.
            valid_ray_mask (tensor): filter corner cases.
            valid_ray_count (tensor): 
        """
        N_surface = self.N_surface_tracker if is_tracker else self.N_surface_mapper
        near_end_surface = self.near_end_surface_tracker if is_tracker else self.near_end_surface_mapper
        far_end_surface = self.far_end_surface_tracker if is_tracker else self.far_end_surface_mapper

        N_rays = rays_o.shape[0]

        if gt_depth is not None:
            # per ray far rendering distance for pixels that have no depth
            # if the max depth is an outlier, it will be very large. Use 5*mean depth instead then.
            far = torch.minimum(
                5*gt_depth.mean(), torch.max(gt_depth*1.2)).repeat(rays_o.shape[0], 1).float()

            if torch.numel(gt_depth) != 0:
                gt_depth = gt_depth.reshape(-1, 1)
            else:
                # handle error, gt_depth is empty
                warnings.warn('tensor gt_depth is empty, info:')
                print('rays_o', rays_o.shape, 'rays_d', rays_d.shape,
                      'gt_depth', gt_depth, 'is_tracker', is_tracker)
                gt_depth = torch.zeros(N_rays, 1, device=device)
        else:
            # render over 10 m when no depth is available at all
            far = 10 * \
                torch.ones((rays_o.shape[0], 1), device=device).float()
            gt_depth = torch.zeros(N_rays, 1, device=device)

        gt_non_zero_mask = gt_depth > 0
        gt_non_zero_mask = gt_non_zero_mask.squeeze(-1)
        mask_rays_near_pcl = torch.ones(
            N_rays, device=device).type(torch.bool)

        gt_non_zero = gt_depth[gt_non_zero_mask]
        gt_depth_surface = gt_non_zero.repeat(
            1, N_surface)

        t_vals_surface = torch.linspace(
            0.0, 1.0, steps=N_surface, device=device)

        z_vals_surface_depth_none_zero = near_end_surface * gt_depth_surface * \
            (1.-t_vals_surface) + far_end_surface * gt_depth_surface * \
            (t_vals_surface)

        z_vals_surface = torch.zeros(
            gt_depth.shape[0], N_surface, device=device)
        z_vals_surface[gt_non_zero_mask,
                       :] = z_vals_surface_depth_none_zero
        if gt_non_zero_mask.sum() < N_rays:
            # determine z_vals_surface values for zero-valued depth pixels
            if self.sample_near_pcl:
                # do ray marching from near_end to far, check if there is a line segment close to point cloud
                # we sample 25 points between near_end and far_end
                # the mask_not_near_pcl is True for rays that are not close to the npc
                z_vals_depth_zero, mask_not_near_pcl = npc.sample_near_pcl(rays_o[~gt_non_zero_mask].detach().clone(),
                                                                           rays_d[~gt_non_zero_mask].detach(
                ).clone(),
                    self.near_end, torch.max(far), N_surface)
                if torch.sum(mask_not_near_pcl.ravel()):
                    # after ray marching, some rays are not close to the point cloud
                    rays_not_near = torch.nonzero(~gt_non_zero_mask, as_tuple=True)[
                        0][mask_not_near_pcl]
                    # update the mask_rays_near_pcl to False for the rays where mask_not_near_pcl is True
                    mask_rays_near_pcl[rays_not_near] = False
                z_vals_surface[~gt_non_zero_mask, :] = z_vals_depth_zero
            else:
                # simply sample uniformly
                z_vals_surface[~gt_non_zero_mask, :] = torch.linspace(self.near_end, torch.max(
                    far), steps=N_surface, device=device).repeat((~gt_non_zero_mask).sum(), 1)

        z_vals = z_vals_surface

        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pointsf = pts.reshape(-1, 3)

        # pointsf.requires_grad_(True)

        ray_pts_num = N_surface
        rays_d_pts = rays_d.repeat_interleave(
            ray_pts_num, dim=0).reshape(-1, 3)
        if self.use_dynamic_radius:
            dynamic_r_query = dynamic_r_query.reshape(
                -1, 1).repeat_interleave(ray_pts_num, dim=0)
        # is_tracker = True
        raw, valid_ray_mask, point_mask, valid_ray_counts = self.eval_points(
            pointsf, decoders, npc, stage, device, npc_geo_feats,
            npc_col_feats, is_tracker, cloud_pos, rays_d_pts,
            ray_pts_num=ray_pts_num, dynamic_r_query=dynamic_r_query,
            exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl)
       
        # y = raw[point_mask,-1]
        # d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # gradients = torch.autograd.grad(
        #     outputs=y,
        #     inputs=pointsf,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        # print(gradients)


        with torch.no_grad():
            raw[torch.nonzero(~point_mask).flatten(), -1] = - 100.0
        raw = raw.reshape(N_rays, ray_pts_num, -1)
        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, device=device, coef=self.sigmoid_coefficient)

        # filter two cases:
        # 1. ray has no gt_depth and it's not close to the current npc
        # 2. ray has gt_depth, but all its sampling locations have no neighbors in current npc
        valid_ray_mask = valid_ray_mask & mask_rays_near_pcl

        if not self.sample_near_pcl:
            depth[~gt_non_zero_mask] = 0
        return depth, uncertainty, color, valid_ray_mask, valid_ray_counts

    @torch.no_grad()
    def render_img(self, npc, decoders, c2w, device, stage, gt_depth=None,
                   npc_geo_feats=None, npc_col_feats=None,
                   dynamic_r_query=None, cloud_pos=None, exposure_feat=None,
                   use_coarse_pcl=False, is_tracker=False):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            npc (): Neural point cloud.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        H = self.H
        W = self.W
        # for all pixels, considering cropped edges
        rays_o, rays_d = get_rays(
            H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
        rays_o = rays_o.reshape(-1, 3)  # (H, W, 3)->(H*W, 3)
        rays_d = rays_d.reshape(-1, 3)
        if self.use_dynamic_radius:
            dynamic_r_query = dynamic_r_query.reshape(-1, 1)

        depth_list = []
        uncertainty_list = []
        color_list = []
        valid_ray_mask_list = []
        valid_ray_count_list = []

        ray_batch_size = self.ray_batch_size

        # run batch by batch
        for i in range(0, rays_d.shape[0], ray_batch_size):
            rays_d_batch = rays_d[i:i+ray_batch_size]
            rays_o_batch = rays_o[i:i+ray_batch_size]
            if gt_depth is None:
                # ret = self.render_batch_ray(
                #     npc, decoders, rays_d_batch, rays_o_batch, device, stage,
                #     gt_depth=None, npc_geo_feats=npc_geo_feats, npc_col_feats=npc_col_feats,
                #     is_tracker=is_tracker,
                #     cloud_pos=cloud_pos,
                #     dynamic_r_query=dynamic_r_query[i:i +
                #                                     ray_batch_size] if self.use_dynamic_radius else None,
                #     exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl)
                ret = self.render_batch_ray_no_depth(
                    npc, decoders, rays_d_batch, rays_o_batch, device, stage,
                    npc_geo_feats=npc_geo_feats, npc_col_feats=npc_col_feats,
                    is_tracker=is_tracker,
                    cloud_pos=cloud_pos,
                    dynamic_r_query=dynamic_r_query[i:i +
                                                    ray_batch_size] if self.use_dynamic_radius else None,
                    exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl)
            else:
                gt_depth = gt_depth.reshape(-1)
                gt_depth_batch = gt_depth[i:i+ray_batch_size]
                ret = self.render_batch_ray(
                    npc, decoders, rays_d_batch, rays_o_batch, device, stage,
                    gt_depth=gt_depth_batch, npc_geo_feats=npc_geo_feats, npc_col_feats=npc_col_feats,
                    is_tracker=is_tracker,
                    cloud_pos=cloud_pos,
                    dynamic_r_query=dynamic_r_query[i:i +
                                                    ray_batch_size] if self.use_dynamic_radius else None,
                    exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl)

            depth, uncertainty, color, valid_ray_mask, valid_ray_count = ret
            depth_list.append(depth.double())
            uncertainty_list.append(uncertainty.double())
            valid_ray_mask_list.append(valid_ray_mask.double())
            valid_ray_count_list.append(valid_ray_count.double())
            # list of tensors here
            color_list.append(color)

        # cat to one big tensor
        depth = torch.cat(depth_list, dim=0)
        uncertainty = torch.cat(uncertainty_list, dim=0)
        color = torch.cat(color_list, dim=0)
        valid_ray_mask = torch.cat(valid_ray_mask_list,dim=0)
        valid_ray_count = torch.cat(valid_ray_count_list, dim=0)

        depth = depth.reshape(H, W)
        uncertainty = uncertainty.reshape(H, W)
        color = color.reshape(H, W, 3)
        valid_ray_mask = valid_ray_mask.reshape(H,W)
        valid_ray_count = valid_ray_count.reshape(H,W)
        return depth, uncertainty, color, valid_ray_mask, valid_ray_count




    def render_batch_ray_no_depth(self, npc, decoders, rays_d, rays_o, device, stage,
                         npc_geo_feats=None, npc_col_feats=None, is_tracker=False, cloud_pos=None,
                         dynamic_r_query=None, exposure_feat=None, use_coarse_pcl=False):
        """
        Render color, depth, uncertainty and a valid ray mask from a batch of rays.

        Args:
            npc (): Neural point cloud.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            is_tracker (bool, optional): tracker has different gradient flow in eval_points.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty (can be interpreted as epistemic uncertainty)
            color (tensor): rendered color.
            valid_ray_mask (tensor): filter corner cases.
        """

        N_rays = rays_o.shape[0]

            # render over 10 m when no depth is available at all
        far = 15 * \
            torch.ones((rays_o.shape[0], 1), device=device).float()

        ray_pts_num = 64

        t_vals_surface = torch.linspace(
            0.0, 1.0, steps=ray_pts_num, device=device)

        z_vals =  far * (t_vals_surface)
  
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pointsf = pts.reshape(-1, 3)

        rays_d_pts = rays_d.repeat_interleave(
            ray_pts_num, dim=0).reshape(-1, 3)
        if self.use_dynamic_radius:
            dynamic_r_query = dynamic_r_query.reshape(
                -1, 1).repeat_interleave(ray_pts_num, dim=0)

        raw, valid_ray_mask, point_mask, valid_ray_counts = self.eval_points(
            pointsf, decoders, npc, stage, device, npc_geo_feats,
            npc_col_feats, is_tracker, cloud_pos, rays_d_pts,
            ray_pts_num=ray_pts_num, dynamic_r_query=dynamic_r_query,
            exposure_feat=exposure_feat, use_coarse_pcl=use_coarse_pcl)

        with torch.no_grad():
            raw[torch.nonzero(~point_mask).flatten(), -1] = - 100.0
        raw = raw.reshape(N_rays, ray_pts_num, -1)
        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, device=device, coef=self.sigmoid_coefficient)

        # filter two cases:
        # 1. ray has no gt_depth and it's not close to the current npc
        # 2. ray has gt_depth, but all its sampling locations have no neighbors in current npc
        valid_ray_mask = valid_ray_mask 

        return depth, uncertainty, color, valid_ray_mask, valid_ray_counts