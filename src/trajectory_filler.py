import torch
import lietorch
from lietorch import SE3
from src.factor_graph import FactorGraph
from tqdm import tqdm
from src.utils import pose_ops


class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """
    def __init__(self, net, video, device='cuda:0'):

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, timestamps, images, depths, intrinsics):
        """ fill operator """
        tt = torch.tensor(timestamps, device=self.device)
        images = torch.stack(images, dim=0)
        if depths is not None:
            depths = torch.stack(depths, dim=0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images.to(self.device)

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(timestamps)

        ts = self.video.timestamp[:N]
        Ps = SE3(self.video.poses[:N])

        # found the location of current timestamp in keyframe queue
        t0 = torch.tensor([ts[ts<=t].shape[0] - 1 for t in timestamps])
        t1 = torch.where(t0 < N-1, t0+1, t0)

        # time interval between nearby keyframes
        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(dim=-1)
        w = v * (tt - ts[t0]).unsqueeze(dim=-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        # temporally put the non-keyframe at the end of keyframe queue
        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:, 0], Gs.data, 1, depths, intrinsics / 8.0, fmap)

        graph = FactorGraph(self.video, self.update)
        # build edge between current frame and nearby keyframes for optimization
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for itr in range(12):
            graph.update(N, N+M, motion_only=True)

        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        return [Gs]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images. """

        # store all camera poses
        pose_list = []

        timestamps = []
        images = []
        intrinsics = []

        # for (timestamp, image, depth, intrinsic, gt_pose) in image_stream:
        print("Filling full trajectory ...")
        for (timestamp, image, depth, intrinsic, gt_pose,_,_)  in tqdm(image_stream):
            timestamps.append(timestamp)
            images.append(image)
            intrinsics.append(intrinsic)

            if len(timestamps) == 16:
                pose_list += self.__fill(timestamps, images, None, intrinsics)
                timestamps, images, depths, intrinsics = [], [], [], []

        if len(timestamps) > 0:
            pose_list += self.__fill(timestamps, images, None, intrinsics)

        # stitch pose segments together
        return lietorch.cat(pose_list, dim=0)


class PoseTrajectoryExtrapolation(PoseTrajectoryFiller):
    """ pose extrapolation using video """
    def __init__(self, net, video, device):
        super().__init__(net, video, device)
        self._ba_window = 3  # TODO move to config

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        return self.fnet(image)

    def __slerp(self, timestamp):
        """ output pose is c2w """
        if self.video.counter.value == 1:
            prev_pose = self.video.get_pose_tensor(0).detach().clone()
            return prev_pose.detach().clone()
        curr_kf_idx = self.video.counter.value - 1
        t0 = self.video.timestamp[curr_kf_idx-1]
        t1 = self.video.timestamp[curr_kf_idx]
        pose0 = self.video.get_pose_tensor(curr_kf_idx-1).detach().clone()
        pose1 = self.video.get_pose_tensor(curr_kf_idx).detach().clone()
        t = (timestamp - t0) / (t1 - t0)
        pose = pose_ops.pose_slerp(pose0, pose1, t)
        return pose

    def __ba(self, pose, timestamp, image, intrinsic):
        """ input and output poses are c2w"""
        # video.poses saves w2c. use video.set_pose_tensor to save c2w

        intrinsic = intrinsic.to(self.device)
        
        inputs = torch.stack([image], dim=0).to(self.device).sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)
        N = self.video.counter.value
        window = min(N, self._ba_window)

        self.video.counter.value += 1
        self.video[N] = (timestamp, image, None, 1, None, intrinsic / 8.0, fmap)
        self.video.set_pose_tensor(N, pose)

        graph = FactorGraph(self.video, self.update, device=self.device)
        t0 = torch.arange(N-window, N).to(self.device)  # [N-window, N-window+1, ..., N-1] len=window
        t1 = torch.arange(N, N+1).to(self.device).expand(window)  # [N, N, ..., N] len=window
        graph.add_factors(t0, t1)

        for _ in range(8):
            graph.update(N, N+1, motion_only=True)

        pose = SE3(self.video.get_pose_tensor(N))
        self.video.counter.value -= 1
        return pose.data

    @torch.no_grad()
    def __call__(self, timestamp, image, intrinsic, do_ba=True):
        pose = self.__slerp(timestamp)
        if do_ba:
            pose = self.__ba(pose, timestamp, image, intrinsic)
        return pose
