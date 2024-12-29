import os

import lietorch
import numpy as np
import torch


class PoseHistory:
    """
    Class to save trajectories of previously known length.
    Timestamps go from 0 to max_time-1.
    It stores up to (max_time,7) poses and (max_time,) timestamps.
    Allows for sparse trajectories.
    """
    def __init__(self, max_time, device="cuda"):
        """ initialize PoseHistory. shared memory tensors are used.
        Args:
            max_time (int): maximum number of timestamps to store, buffer size.
            device (str): device to store.
        """
        self._max_time = max_time
        self.poses = torch.zeros((max_time, 7), dtype=torch.float, device=device)
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses.share_memory_()
        self.times = torch.zeros((max_time,), dtype=torch.float, device=device)
        self.times[:] = -1
        self.times.share_memory_()
        self.is_set = torch.zeros((max_time,), dtype=torch.bool, device=device)
        self.is_set.share_memory_()

        self._last_time = -1

    def append(self, time, pose):
        """ append pose to history with timestamp. fails if past poses are inserted.
        Args:
            time (int/float/torch.tensor): timestamp/index
            pose (torch.tensor): (7,) shape pose
        """
        if time >= self._max_time:
            raise ValueError(f"Time {time} is greater than max_time {self._max_time}")
        if time <= self._last_time:
            raise ValueError(f"Time {time} is smaller than last time {self._last_time}")
        
        idx = int(time)
        self.is_set[idx] = True
        self.times[idx] = time.item() if type(time) == torch.Tensor else time
        self.poses[idx, :] = pose.reshape(-1)

        self._last_time = time

    def __getitem__(self, idx):
        """ get pose at history index.
        Args:
            idx (int/slice): index(es)
        Returns:
            time (float/np.array): timestamp(s)
            poses (torch.tensor): (7,) shape pose(s)
        """
        nonull_idx = torch.nonzero(self.times >= 0).squeeze(-1)
        idx = nonull_idx[idx]  # index in non-null timestamps, allows negative indices and slicing
        time = self.times[idx]
        if type(idx) == int:
            time = time.item()
        else:  # slice
            time = time.cpu().numpy()
        pose = self.poses[idx].clone()
        return time, pose
    
    def get_pose_at_time(self, time):
        """ get pose at time
        Args:
            time (int/float): timestamp
        Returns:
            pose (torch.tensor): (7,) shape pose
        """
        idx = int(time)
        if not self.is_set[idx]:
            return None
        return self.poses[idx].clone()


    def get_trajectory(self, matrix=False, numpy=False):
        """ get all poses and times in the trajectory.
        Returns:
            times (torch.tensor): (N,) timestamps
            poses (torch.tensor): (N,7) poses
            matrix (bool): if True, return poses in matrix form (N,4,4). This assumes poses are in format [tx,ty,tz,qx,qy,qz,qw].
            numpy (bool): if True, return poses and times in numpy form.
        """
        nonull = self.times >= 0
        times = self.times[nonull].clone()
        poses = self.poses[nonull].clone()
        if matrix:
            poses = lietorch.SE3(poses).matrix().data  # (N,7) -> (N,4,4)
        if numpy:
            times = times.cpu().numpy()
            poses = poses.cpu().numpy()
        return times, poses
    
    def __len__(self):
        """ get number of timestamps filled.
        Returns:
            length (int): number of timestamps filled.
        """
        return torch.sum(self.times >= 0).item()

    def is_full(self):
        """ check if trajectory is full.
        Returns:
            full (bool): True if all timestamps are filled.
        """
        return torch.all(self.times >= 0)

    def save(self, path):
        """ save pose history to npz file.
        Args:
            path (str): path to save.
        """
        np.savez(path, poses=self.poses.cpu().numpy(), timestamps=self.times.cpu().numpy())


class OutputTrajectory:
    def __init__(self, est_times, est_poses, is_full=False):
        """ represents an output trajectory.
        generated from:
            - from_traj_filler: uses traj_filler to get full interpolated estimation. uses images from stream
            - from_pose_history: uses pose_history with arbitrary timestamps and poses history.
        Attributes:
            est_times (np.array): (N,) timestamps
            est_poses (np.array): (N,4,4) poses, representing c2w matrices
            is_full (bool): if True, it means trajectory is full
        """
        self._est_times = est_times
        self._est_poses = est_poses
        self._is_full = is_full

    @staticmethod
    def from_traj_filler(traj_filler, stream):
        traj_est_inv = traj_filler(stream)  # w2c lie, full interpolated estimation. uses images from stream
        traj_est_lietorch = traj_est_inv.inv()  # c2w lie
        traj_est = traj_est_lietorch.matrix().data.cpu().numpy()  # c2w matrix numpy
        kf_num = traj_filler.video.counter.value
        kf_timestamps = traj_filler.video.timestamp[:kf_num].cpu().int().numpy()
        kf_poses = lietorch.SE3(traj_filler.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
        traj_est[kf_timestamps] = kf_poses  # fix kf poses
        timestamps = np.arange(traj_est.shape[0], dtype=np.float32)
        return OutputTrajectory(timestamps, traj_est, is_full=True)
    
    @staticmethod
    def from_pose_history(pose_history: PoseHistory):
        times, poses = pose_history.get_trajectory(matrix=True, numpy=True)
        return OutputTrajectory(times, poses, is_full=pose_history.is_full())
    
    @staticmethod
    def from_video_npz(npz_path):
        offline_video = dict(np.load(npz_path))
        video_traj = offline_video['poses']  # c2w matrix
        video_timestamps = offline_video['timestamps']
        # not full, since it only contains keyframes
        return OutputTrajectory(video_timestamps, video_traj)
    
    def get_trajectory(self):
        return self._est_times, self._est_poses
    
    def is_full(self):
        return self._is_full


class PsnrExpertHist:
    """save all expert online predictions"""
    def __init__(self):
        self.timestamp_hist = []
        self.gt_pose_hist = []
        self.x1_pose_hist = []
        self.x2_pose_hist = []
        self.x3_pose_hist = []
        self.x4_pose_hist = []
        self.x5_pose_hist = []
        self.x1_psnr_hist = []
        self.x2_psnr_hist = []
        self.x3_psnr_hist = []
        self.x4_psnr_hist = []
        self.x5_psnr_hist = []
        self.x1_loss_hist = []
        self.x2_loss_hist = []
        self.x3_loss_hist = []
        self.x4_loss_hist = []
        self.x5_loss_hist = []

    def append_expert(self, expert_idx, pose, psnr, loss):
        hist = [self.x1_pose_hist, self.x2_pose_hist, self.x3_pose_hist, self.x4_pose_hist, self.x5_pose_hist][expert_idx]
        hist.append(pose.detach().cpu().numpy())
        hist = [self.x1_psnr_hist, self.x2_psnr_hist, self.x3_psnr_hist, self.x4_psnr_hist, self.x5_psnr_hist][expert_idx]
        if isinstance(psnr, torch.Tensor):
            psnr = psnr.item()
        hist.append(psnr)
        hist = [self.x1_loss_hist, self.x2_loss_hist, self.x3_loss_hist, self.x4_loss_hist, self.x5_loss_hist][expert_idx]
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        hist.append(loss)

    def append_gt(self, timestamp, pose):
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
        self.timestamp_hist.append(timestamp)
        self.gt_pose_hist.append(pose.detach().cpu().numpy())

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        np.savez(os.path.join(dir_path, "psnr_expert_hist.npz"),
                 timestamp_hist=self.timestamp_hist,
                 gt_pose_hist=self.gt_pose_hist,
                 x1_pose_hist=self.x1_pose_hist,
                 x2_pose_hist=self.x2_pose_hist,
                 x3_pose_hist=self.x3_pose_hist,
                 x4_pose_hist=self.x4_pose_hist,
                 x5_pose_hist=self.x5_pose_hist,
                 x1_psnr_hist=self.x1_psnr_hist,
                 x2_psnr_hist=self.x2_psnr_hist,
                 x3_psnr_hist=self.x3_psnr_hist,
                 x4_psnr_hist=self.x4_psnr_hist,
                 x5_psnr_hist=self.x5_psnr_hist,
                 x1_loss_hist=self.x1_loss_hist,
                 x2_loss_hist=self.x2_loss_hist,
                 x3_loss_hist=self.x3_loss_hist,
                 x4_loss_hist=self.x4_loss_hist,
                 x5_loss_hist=self.x5_loss_hist)
