import torch
import lietorch
from src.utils.pose_ops import pose_loss, pose_loss_TR, project_to_subspace, pose_err_T, pose_err_R


# error metrics

def pose_dist(pose1, pose2):
    return pose_loss(pose1, pose2).item()


def pose_dist_TR(pose1, pose2):
    W_ROT = 1.0  # TODO: add to config
    return pose_loss_TR(pose1, pose2, W_ROT).item()


def pose_err_t(pose1, pose2):
    return pose_err_T(pose1[:3], pose2[:3]).item()


def pose_err_r(pose1, pose2):
    return pose_err_R(pose1[3:], pose2[3:]).item()


def cosine_similarity(pose1, pose2):
    """cosine similarity between two poses"""
    v1 = lietorch.SE3(pose1).log()
    v2 = lietorch.SE3(pose2).log()
    cossim = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-6)
    return cossim.item()


def norm_ratio(pose1, pose2):
    """||pose1|| / ||pose2||"""
    v1 = lietorch.SE3(pose1).log()
    v2 = lietorch.SE3(pose2).log()
    return (v1.norm() / (v2.norm() + 1e-6)).item()


def dist_to_plane(pose, u1, u2):
    proj = project_to_subspace(pose, u1, u2)
    return torch.norm(pose - proj).item()


# pose error evaluator

class PoseErrorEvaler:
    """ online evaluation of APE and RPateE for a single pose """ 

    def __init__(self):
        self._metrics = {}
        self._print_metrics = {}  # only for printing

    @staticmethod
    def _compute_ape(est_pose, gt_pose):
        """ Absolute Pose Error for a single pose.

        Args:
            est_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, estimated pose
            gt_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, ground truth pose

        Returns:
            ape_full: float, full APE
            ape_pos: float, position APE
            ape_rot: float, rotation APE
            ape_deg: float, rotation angle APE in degrees
        """
        # implementation based on evo.core.metrics

        relative_lie = lietorch.SE3(est_pose).inv() * lietorch.SE3(gt_pose)
        relative_mat = relative_lie.matrix().data

        device = est_pose.device
        ape_full = torch.norm(relative_mat - torch.eye(4, device=device)).item()
        ape_pos = torch.norm(est_pose[:3] - gt_pose[:3]).item()
        ape_rot = torch.norm(relative_mat[:3, :3] - torch.eye(3, device=device)).item()
        ape_deg = lietorch.SO3(relative_lie).log().norm().item()

        return ape_full, ape_pos, ape_rot, ape_deg

    @staticmethod
    def _compute_rpe(curr_est_pose, curr_gt_pose, prev_est_pose, prev_gt_pose):
        """ Relative Pose Error for a single pose update.

        Args:
            curr_est_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, estimated pose in current timestamp
            curr_gt_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, ground truth pose in current timestamp
            prev_est_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, estimated pose in previous timestamp
            prev_gt_pose: torch.Tensor of shape=(7,) or lietorch.SE3 instance, ground truth pose in previous timestamp

        Returns:
            rpe_full: float, full RPE
            rpe_pos: float, position RPE
            rpe_rot: float, rotation RPE
            rpe_deg: float, rotation angle RPE in degrees
        """
        delta_est_pose = lietorch.SE3(prev_est_pose).inv() * lietorch.SE3(curr_est_pose)
        delta_gt_pose = lietorch.SE3(prev_gt_pose).inv() * lietorch.SE3(curr_gt_pose)
        return PoseErrorEvaler._compute_ape(delta_est_pose.data, delta_gt_pose.data)

    def eval_pe(self, curr_est_pose, curr_gt_pose, prev_est_pose, prev_gt_pose, name_prefix):
        """ compute APE and RPE metrics for a single pose step.
        inputs are c2w poses in (7,) tensor format [tx, ty, tz, qx, qy, qz, qw]"""

        if curr_est_pose is None or curr_gt_pose is None:
            return

        # compute APE metrics
        ape, ape_pos, ape_rot, ape_deg = self._compute_ape(curr_est_pose, curr_gt_pose)

        self._metrics[f'{name_prefix}_APE'] = ape
        self._metrics[f'{name_prefix}_APE-pos'] = ape_pos
        self._metrics[f'{name_prefix}_APE-rot'] = ape_rot
        self._metrics[f'{name_prefix}_APE-deg'] = ape_deg

        if prev_est_pose is not None and prev_gt_pose is not None:
            # compute RPE metrics
            rpe, rpe_pos, rpe_rot, rpe_deg = self._compute_rpe(curr_est_pose, curr_gt_pose, prev_est_pose, prev_gt_pose)

            self._metrics[f'{name_prefix}_RPE'] = rpe
            self._metrics[f'{name_prefix}_RPE-pos'] = rpe_pos
            self._metrics[f'{name_prefix}_RPE-rot'] = rpe_rot
            self._metrics[f'{name_prefix}_RPE-deg'] = rpe_deg

    def eval_deltas(self, est_delta, gt_delta, name_prefix):
        """ compute additional metrics for a single pose step."""
        if est_delta is None or gt_delta is None:
            return
        self._metrics[f'{name_prefix}_dist'] = pose_dist(est_delta, gt_delta)
        self._print_metrics[f'{name_prefix}_dist_TR'] = pose_dist_TR(est_delta, gt_delta)
        self._metrics[f'{name_prefix}_err_T'] = pose_err_t(est_delta, gt_delta)
        self._metrics[f'{name_prefix}_err_R'] = pose_err_r(est_delta, gt_delta)
        self._metrics[f'{name_prefix}_cossim'] = cosine_similarity(est_delta, gt_delta)
        self._metrics[f'{name_prefix}_norm'] = norm_ratio(est_delta, gt_delta)
    
    def compare_deltas(self, delta1, delta2, name_prefix):
        if delta1 is None or delta2 is None:
            return
        self._metrics[f'{name_prefix}_dist'] = pose_dist(delta1, delta2)
        self._print_metrics[f'{name_prefix}_dist_TR'] = pose_dist_TR(delta1, delta2)
        self._print_metrics[f'{name_prefix}_err_T'] = pose_err_t(delta1, delta2)
        self._print_metrics[f'{name_prefix}_err_R'] = pose_err_r(delta1, delta2)
        self._metrics[f'{name_prefix}_cossim'] = cosine_similarity(delta1, delta2)
        self._metrics[f'{name_prefix}_norm'] = norm_ratio(delta1, delta2)

    def dist_to_plane(self, target_delta, u1_delta, u2_delta, name_prefix):
        if target_delta is None or u1_delta is None or u2_delta is None:
            return
        self._metrics[f'{name_prefix}_dist_to_plane'] = dist_to_plane(target_delta, u1_delta, u2_delta)

    def get_metrics(self):
        return self._metrics
    
    def print_metrics(self, fn_logger=None):
        if fn_logger is None:
            fn_logger = print
        fn_logger('Pose Error Metrics:')
        for k, v in self._metrics.items():
            fn_logger(f'\t{k}: {v:.4f}')
        fn_logger('\t---')
        for k, v in self._print_metrics.items():
            fn_logger(f'\t{k}: {v:.4f}')

    def reset(self):
        self._metrics = {}
