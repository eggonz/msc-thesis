import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict
from src.common import get_bound_from_pointcloud, in_bound

from src.droid_net import cvx_upsample
import src.geom.projective_ops as pops

class DepthVideo:
    def __init__(self, cfg, args):
        self.cfg =cfg
        self.args = args
        self.offline_mapping = cfg["offline_mapping"]
        self.output = cfg['data']['output']
        # current keyframe count
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        buffer = cfg['tracking']['buffer']
        self.enable_depth_prior = cfg['tracking']['backend']['enable_depth_prior']
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = args.device

        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.wq_dirty = torch.ones(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.npc_dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()  # upsample(disps) in mono, gt_depth in RGBD
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.bool).share_memory_()        
        self.already_opt = -1

        self.stereo = False
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        ### pose compensation from vitural to real
        self.pose_compensate = torch.zeros(1, 7, dtype=torch.float, device=self.device).share_memory_()
        self.pose_compensate[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)

        if self.offline_mapping:
            # self.graph_path = f"{self.output}/{self.cfg['offline_graph']}"
            self.video_path = f"{self.output}/{self.cfg['offline_video']}"
            offline_video = np.load(self.video_path)
            self.offline_video = {'depths':torch.from_numpy(offline_video['depths']),
                                  'poses':torch.from_numpy(offline_video['poses']),
                                  'scale':offline_video['scale'],
                                  'valid_depth_masks': torch.from_numpy(offline_video['valid_depth_masks'])}

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            mono_depth = item[4][3::8,3::8]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            # mono_depth_up = item[4]
            # self.disps_up[index] = torch.where(mono_depth_up>0, 1.0/mono_depth_up, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba_only(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, ba_type="ba"):
        """ dense bundle adjustment (DBA) 
        Computes and returns updated poses and disps. """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            if ba_type == "ba":
                target = target.view(-1, self.ht//8, self.wd//8, 2).permute(0,3,1,2).contiguous()
                weight = weight.view(-1, self.ht//8, self.wd//8, 2).permute(0,3,1,2).contiguous()
                poses, disps = self.poses.clone(), self.disps.clone()  # shapes: (400, 7[3transl&4quatern]), (400, 40[320//8], 80[640//8])
                # poses, disps updated inplace
                droid_backends.ba(poses, disps, self.intrinsics[0], self.mono_disps,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)  # HERE compute GN, inplace update poses+disps. cuda code.
                disps.clamp_(min=1e-5)
                return True, poses, disps

            elif ba_type == "wq_ba":
                # return False
                poses = lietorch.SE3(self.poses[None])
                disps = self.disps[None]
                scales = self.depth_scale
                shifts = self.depth_shift
                ignore_frames = 0
                self.update_valid_depth_mask_small()
                curr_idx = self.counter.value-1
                mono_d = self.mono_disps[:curr_idx+1]
                est_d = self.disps[:curr_idx+1]
                valid_d = self.valid_depth_mask_small[:curr_idx+1]
                scale_t, shift_t, error_t = scale_shift_error(mono_d, est_d, valid_d)
                avg_disps = est_d.mean(dim=[1,2])
                # print("avg_disps",avg_disps)

                scales[:curr_idx+1]=scale_t
                shifts[:curr_idx+1]=shift_t

                target_t,weight_t,eta_t,ii_t,jj_t = target,weight,eta,ii,jj

                ################################################################
                if self.mono_thres:
                    # remove the edges which contains poses with invalid mono depth
                    invalid_mono_mask = (error_t/avg_disps > self.mono_thres)| \
                                        (error_t.isnan())|\
                                        (scale_t < 0)|\
                                        (valid_d.sum(dim=[1,2]) < 
                                        valid_d.shape[1]*valid_d.shape[2]*0.5)
                    invalid_mono_index, = torch.where(invalid_mono_mask.clone())
                    invalid_ii_mask = (ii<0)
                    idx_in_ii = torch.unique(ii)
                    valid_eta_mask = (idx_in_ii >= 0)
                    for idx in invalid_mono_index:
                        invalid_ii_mask =  invalid_ii_mask | (ii == idx) | (jj == idx) 
                    target_t = target[:,~invalid_ii_mask]
                    weight_t = weight[:,~invalid_ii_mask]
                    ii_t = ii[~invalid_ii_mask]
                    jj_t = jj[~invalid_ii_mask]
                    idx_in_ii_t = torch.unique(ii_t)
                    # print(idx_in_ii_t)
                    valid_eta_mask = torch.tensor([idx in idx_in_ii_t for idx in idx_in_ii]).to(self.device)
                    eta_t = eta[valid_eta_mask]
                ################################################################
                success = False
                for _ in range(itrs):
                    if self.counter.value > ignore_frames and ii_t.shape[0]>0:
                        poses, disps, wqs = src.geom.ba.BA_wq(target_t,weight_t,eta_t,poses,disps,
                                                    self.intrinsics[None],ii_t,jj_t,
                                                    self.mono_disps[None],
                                                    scales[None],shifts[None],
                                                    self.valid_depth_mask_small[None], ignore_frames,
                                                    lm,ep,alpha=0.01)
                        scales = wqs[0,:,0]
                        shifts = wqs[0,:,1]
                        success = True                    

                self.depth_scale = scales
                self.depth_shift = shifts

                disps = disps.squeeze(0)
                poses = poses.vec().squeeze(0)

                disps.clamp_(min=1e-5)
                return success, poses, disps

            else:
                raise NotImplementedError

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, ba_type="ba"):  # TODO change return
        """ Computes and updates poses and disps inplace """
        if self.enable_depth_prior:
            # self.ba_and_update_wq(target, weight, eta, ii, jj, t0, t1, itrs=5, lm=lm, ep=ep)
            success, poses, disps = self.ba_only(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,ba_type)  # HERE (ba_type=ba)
            if not success:
                _ , poses, disps = self.ba_only(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"ba")
        else:
             _ , poses, disps = self.ba_only(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"ba")

        self.poses[:] = poses
        self.disps[:] = disps
            
    
    def get_weight_offline(self,index,device):
        offline_graph = np.load(self.graph_path)
        graph_idx = (offline_graph['ii']==index)
        weight = torch.from_numpy(offline_graph['weight_up'][0,graph_idx]).to(device)
        # weight:torch.Tensor = (offline_graph['weight_up'][graph_idx]).to(device) #[n,h,w,2]
        weight = weight.norm(p=2,dim=3) #[n,h,w]

        weight[weight<0.9] = 0

        weight = weight.mean(dim=0) #[h,w]

        return weight

    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        # mono_depth: [B, n]
        # weights: [B, n]
        if weights is None:
            weights = torch.ones_like(mono_depth).to(mono_depth.device)
        n = mono_depth.shape[-1]
        mono_depth = mono_depth.reshape(-1,n)
        est_depth = est_depth.reshape(-1,n)
        weights = weights.reshape(-1,n)

        # if torch.any(self.wq_dirty[index]):
        if True:
        # if self.wq_dirty[index]:
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

            self.depth_scale[index] = scale
            self.depth_shift[index] = shift
            self.wq_dirty[index] = False
            return [self.depth_scale[index], self.depth_shift[index]]
        else:
            return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device='cuda:0'):
        if self.offline_mapping:
            c2w = self.offline_video['poses'][index].clone().to(device)
        else:
            w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
            c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(device) * w2c.inv()
            c2w = c2w.matrix().clone()  # [4, 4]
        return c2w
    
    def get_pose_tensor(self, index, device='cuda:0'):
        w2c_lie = lietorch.SE3(self.poses[index].clone()).to(device)
        w2c_comp = lietorch.SE3(self.pose_compensate[0].clone()).to(device)
        c2w_lie = w2c_comp * w2c_lie.inv()
        return c2w_lie.data.clone()
    
    def set_pose_tensor(self, index, pose, device='cuda:0'):
        c2w_lie = lietorch.SE3(pose.clone()).to(device)
        w2c_comp = lietorch.SE3(self.pose_compensate[0].clone()).to(device)
        # c2w_lie = w2c_comp * w2c_lie.inv()
        # w2c_comp.inv() * c2w_lie = w2c_comp.inv() * w2c_comp * w2c_lie.inv() = w2c_lie.inv()
        # (w2c_comp.inv() * c2w_lie).inv() = w2c_lie.inv().inv() = w2c_lie
        w2c_lie = (w2c_comp.inv() * c2w_lie).inv()
        self.poses[index,:] = w2c_lie.data.clone()

    def get_depth_and_pose(self, index, device='cuda:0'):
        if self.offline_mapping:
            est_depth = self.offline_video['depths'][index].clone().to(device)
            c2w = self.offline_video['poses'][index].clone().to(device)
            scale = float(self.offline_video['scale'])
            depth_mask = self.offline_video['valid_depth_masks'][index].clone().to(device)
            # est_depth[~depth_mask] = 0
            return est_depth, depth_mask, c2w, scale
        else:
            scale = 1.0
            with self.get_lock():
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                # est_depth[~depth_mask] = 0
                # origin alignment
                w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
                c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(w2c.device) * w2c.inv()
                c2w = c2w.matrix()  # [4, 4]
            return est_depth, depth_mask, c2w, scale

    def get_first_pose(self):
        # self.poses[0] is always frozen to Identity
        w2c = lietorch.SE3(self.poses[0].clone()).to(self.device) # Tw(droid)_to_c
        c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(w2c.device) * w2c.inv()
        c2w = c2w.matrix()  # [4, 4]
        return c2w

    @torch.no_grad()
    def update_valid_depth_mask(self):
        with self.get_lock():
            t = self.counter.value 
            dirty_index, = torch.where(self.dirty.clone())

        if len(dirty_index) == 0:
            return

        # convert poses to 4x4 matrix
        disps_up = torch.index_select(self.disps_up, 0, dirty_index)
        poses = torch.index_select(self.poses, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * 8.0
        depths_up = 1.0/disps_up
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths_up.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up, intrinsic, dirty_index, thresh)  # FIXME single frame (t=0) will fail: no counts.

        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 

        depths_up[~multiview_masks]=torch.nan
        depths_up_reshape = depths_up.view(depths_up.shape[0],-1)
        depths_median = depths_up_reshape.nanmedian(dim=1).values
        masks = depths_up < 3*depths_median[:,None,None]

        kernel_size = self.cfg['tracking']['multiview_filter']['kernel_size']
        if kernel_size == 1:
            self.valid_depth_mask[dirty_index] = masks 
        elif kernel_size == 'inf':
            if masks.sum() < 100:
                return
            points = droid_backends.iproj((lietorch.SE3(poses).inv()).data, disps_up, intrinsic).cpu()    
            sel_points = points.reshape(-1, 3)[masks.reshape(-1)]
            bound = get_bound_from_pointcloud(sel_points) # [3, 2]
            extended_masks = torch.ones_like(masks,device="cpu").bool()
            sel_points = points.reshape(-1, 3)[extended_masks.reshape(-1)]
            in_bound_mask = in_bound(sel_points, bound)  # N'
            extended_masks[extended_masks.clone()] = in_bound_mask
            self.valid_depth_mask[dirty_index] = extended_masks.to(self.device)
        else:
            raise NotImplementedError

        self.dirty[dirty_index] = False
   
    
    
    @torch.no_grad()
    def update_valid_depth_mask_small(self):
        curr_idx = self.counter.value-1
        dirty_index = torch.arange(curr_idx+1).to(self.device)
        disps = torch.index_select(self.disps, 0, dirty_index)
        intrinsic = self.intrinsics[0].detach()
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps, intrinsic, dirty_index, thresh)        
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True
        self.wq_dirty[index_start:index_end] = True


    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose, _ = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        print(f"Saved final depth video: {path}")


@torch.no_grad()
def scale_shift_error(prediction, target, valid_mask):
    weights = torch.ones_like(target,device=target.device) * valid_mask
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*valid_mask
    error_sum = masked_error.sum(dim=[1,2])
    error_num = valid_mask.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error
