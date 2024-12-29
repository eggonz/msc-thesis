import torch
import lietorch
import numpy as np

from lietorch import SE3
from src.factor_graph import FactorGraph
from src.droid_backend import DroidBackend as LoopClosing
from src.trajectory_filler import PoseTrajectoryExtrapolation


class DroidFrontend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.update_op = net.update  # droid update operator (f2f)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = cfg['tracking']['max_age']
        self.iters1 = 4*2
        self.iters2 = 2*2

        self.warmup = cfg['tracking']['warmup']
        self.beta = cfg['tracking']['beta']
        self.frontend_nms = cfg['tracking']['frontend']['nms']
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh']
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.frontend_thresh = cfg['tracking']['frontend']['thresh']
        self.frontend_radius = cfg['tracking']['frontend']['radius']
        self.upsample = cfg['tracking']['upsample']
        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors']

        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, args, cfg)

        self.graph = FactorGraph(
            video, net.update,
            device=args.device,
            corr_impl='volume',
            max_factors=self.frontend_max_factors,
            upsample=self.upsample
        )

        self.cfg_only_kf = cfg['tracking']['f2f']['only_kf']
        self.cfg_filling = cfg['tracking']['f2f']['pose_filling']
        self.cfg_wait_until_warmup = cfg['tracking']['f2f']['wait_until_warmup']

        self.f2f_extrapol = PoseTrajectoryExtrapolation(net, video, video.device)
        

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), # proximity factors only from last 5 frames
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.mono_disps[self.t1-1] > 0, 
           self.video.mono_disps[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            ba_type = "ba" if itr%2==0 else "wq_ba"
            self.graph.update(None, None, use_inactive=True, ba_type=ba_type)  # t0=None, t1=None (opt all)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        # d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)
        d = self.video.distance([self.t1-2], [self.t1-1], beta=self.beta, bidirectional=True)


        if d.item() < self.keyframe_thresh:
            # self.graph.rm_keyframe(self.t1 - 2)
            self.graph.rm_keyframe(self.t1 - 1)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        # else:
        #     for itr in range(self.iters2):
        #         self.graph.update(None, None, use_inactive=True)
        else:
            cur_t = self.video.counter.value
            t_start = 0
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, 
                                                         motion_only=False, local_graph=self.graph,
                                                         enable_wq=True)
                if n_edge == 0:
                    for itr in range(self.iters2):
                        ba_type = "ba" if itr%2==0 else "wq_ba"
                        self.graph.update(t0=None, t1=None, use_inactive=True,ba_type=ba_type)  # t0=None, t1=None (opt all)
                self.last_loop_t = cur_t
            else:
                for itr in range(self.iters2):
                    ba_type = "ba" if itr%2==0 else "wq_ba"
                    self.graph.update(t0=None, t1=None, use_inactive=True,ba_type=ba_type)  # t0=None, t1=None (opt all)

        # set pose for next iteration
        # # FIXME init this pose before update, not init next pose after update
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()  # TODO remove (done by pose initializer in main tracking loop)  # FIXME fail when buffer size exceeded
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.set_dirty(self.graph.ii.min(), self.t1)
        # self.video.dirty[self.graph.ii.min():self.t1] = True
        torch.cuda.empty_cache()

    def first_frame(self):
        """ call for first frame when cvideo.counter==1 """

        if self.video.counter.value != 1:
            raise ValueError("First frame should be called when video.counter==1")
        
        self.video.poses[1] = self.video.poses[0].clone()  # TODO remove (done by pose initializer in main tracking loop)
        self.video.disps[1] = self.video.disps[0].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.set_dirty(0, 1)

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        if self.t1 > 1:
            self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

            for itr in range(8):
                self.graph.update(1, use_inactive=True,ba_type="ba")  # t0=1, t1=None (freeze first pose)

            self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

            for itr in range(8):
                self.graph.update(1, use_inactive=True,ba_type="ba")  # t0=1, t1=None (freeze first pose)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()  # TODO remove (done by pose initializer in main tracking loop)
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        # self.last_pose = self.video.poses[self.t1-1].clone()
        # self.last_disp = self.video.disps[self.t1-1].clone()
        # self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            # self.video.dirty[:self.t1] = True
            self.video.set_dirty(0, self.t1)

        if self.t1 > 1:
            self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self, timestamp, image, intrinsic):
        """ main update """

        pose = None
        do_extrapolation = False

        # first frame (pose fixed to Id)
        if timestamp == 0:
            self.first_frame()
            pose = self.video.get_pose_tensor(self.video.counter.value - 1)  # Id

        # do initialization (once), when video.counter matches warmup
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            pose = self.video.get_pose_tensor(self.video.counter.value - 1)
            if pose.isnan().any():
                do_extrapolation = True
            
        # do update, when it is new KF
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
            pose = self.video.get_pose_tensor(self.video.counter.value - 1)
            if pose.isnan().any():
                do_extrapolation = True

        # non-KF
        elif timestamp > 0 and not self.cfg_only_kf:
            do_extrapolation = True

        if self.cfg_wait_until_warmup and not self.is_initialized:
            do_extrapolation = False
            pose = None

        if do_extrapolation:
            do_ba = self.cfg_filling == 'slerp_ba'
            pose = self.f2f_extrapol(timestamp, image, intrinsic, do_ba=do_ba)

        if pose is not None and pose.isnan().any():
            if do_extrapolation:
                raise ValueError("F2F pose is still NaN after extrapolation at timestamp: {}".format(timestamp))
            else:
                raise ValueError("F2F pose is NaN at timestamp: {}".format(timestamp))

        self.video.update_valid_depth_mask()

        return pose.detach().clone() if pose is not None else None
