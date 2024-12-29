import torch
from lietorch import SO3, SE3


# SE(3) pose losses

def pose_loss(est_pose, gt_pose):
    """||Log(Exp(est).inv() * Exp(gt)||"""
    if len(est_pose.shape) == 1:
        est_pose = est_pose.unsqueeze(0)  # (B, 7)
        gt_pose = gt_pose.unsqueeze(0)
    est = SE3(est_pose)
    gt = SE3(gt_pose)
    d = est * gt.inv()
    d_log = d.log()
    loss = d_log.norm(dim=-1).mean()
    return loss


def pose_err_T(est_xyz, gt_xyz):
    """||T_est - T_gt||"""
    return torch.norm(est_xyz - gt_xyz, dim=-1)  # ([B,] 3)


def pose_err_R(est_quat, gt_quat):
    """||Log(R_est.inv() * R_gt)||"""
    est_R = SO3(est_quat)  # (B, 4)
    gt_R = SO3(gt_quat)
    d_r = (est_R.inv() * gt_R).log()  # ([B,] 3)
    return d_r.norm(dim=-1)


def pose_loss_TR(est_pose, gt_pose, w_rot=1.0):
    """||T_est - T_gt|| + w_rot * ||Log(R_est.inv() * R_gt)||"""
    if len(est_pose.shape) == 1:
        est_pose = est_pose.unsqueeze(0)  # (B, 7)
        gt_pose = gt_pose.unsqueeze(0)
    err_T = pose_err_T(est_pose[:, :3], gt_pose[:, :3]).mean()
    err_R = pose_err_R(est_pose[:, 3:], gt_pose[:, 3:]).mean()
    loss = err_T + w_rot * err_R
    return loss


# linear operations in R6

def pose_slerp(pose0, pose1, t):
    """ poses in format [tx, ty, tz, qx, qy, qz, qw] """
    p0 = SE3(pose0.float())
    p1 = SE3(pose1.float())
    dp = p1 * p0.inv()
    out = SE3.exp(dp.log() * t) * p0
    return out.data


def pose_mean(pose0, pose1):
    """ pose_slerp(pose0, pose1, t=0.5) """
    return pose_slerp(pose0, pose1, 0.5)


def pose_sum(pose1, pose2):
    """ poses in format [tx, ty, tz, qx, qy, qz, qw]. returns p2 @ p1. shape (7,) or (B,7) """
    p1 = SE3(pose1.float())
    p2 = SE3(pose2.float())
    out = p2 * p1
    return out.data


def pose_sub(pose1, pose2):
    """ poses in format [tx, ty, tz, qx, qy, qz, qw]. returns p2.inv() @ p1. shape (7,) or (B,7) """
    p1 = SE3(pose1.float())
    p2 = SE3(pose2.float())
    out = p2.inv() * p1
    return out.data


def pose_scale(pose, scalar):
    """ poses in format [tx, ty, tz, qx, qy, qz, qw]. returns exp(scalar * p.log()). shape (7,) or (B,7) """
    p = SE3(pose.float())
    out = SE3.exp(scalar * p.log())
    return out.data


# projection in R6

def project_to_subspace(original_pose, *subsp_poses):
    """
    Computes the projection of original_pose on the subspace spanned by subsp_poses in the Lie algebra space,
    using the Gram-Schmidt orthogonalization method.

    Poses are given in format [tx, ty, tz, qx, qy, qz, qw]=R⁷
    Poses are converted to SE(3) group and then to se(3)=R⁶ algebra.
    The linear combination is done in the R⁶ space and then converted back to R⁷.

    Args:
        original_pose: tensor of shape (7,), target pose
        *subsp_poses: tensors of shape (7,), plane poses

    Returns:
        tensor of shape (7,), closest pose to original_pose on the plane defined by plane_poses
    """
    # R⁷ to R⁶
    vs_subsp = [SE3(p).log() for p in subsp_poses]
    v_target = SE3(original_pose).log()
    # projection in R⁶
    projected = sum([torch.dot(v_target, v) * v / torch.dot(v, v) for v in vs_subsp])
    # R⁶ to R⁷
    return SE3.exp(projected).data


# linear combinations in R6

def pose_linear_comb_scalar(pose1, pose2, w1, w2):
    """
    poses in format [tx, ty, tz, qx, qy, qz, qw]
    optional batch dimension

    Args:
        pose1: tensor of shape (7,) or (B,7)
        pose2: tensor of shape (7,) or (B,7)
        w1: scalar or tensor of shape one of (,) (B,) (1,) (B,1)
        w2: scalar or tensor of shape one of (,) (B,) (1,) (B,1)

    Returns:
        out: tensor of shape (7,) or (B,7)
    """
    assert pose1.shape == pose2.shape, 'poses have different shapes'
    assert w1.shape == w2.shape, 'weights have different shapes'
    if len(w1.shape) > 0 and w1.shape[0] != 1:
        assert len(pose1.shape) == 2, 'w has batch dimension but p does not'
        assert w1.shape[0] == pose1.shape[0], 'batch dimension mismatch'
    
    # [,] -> [1,]
    # [B,] -> [B,1]
    # [B=1,] -> [1,]
    # [1,] -> [1,]
    # [B,1] -> [B,1]
    # [B=1,1] -> [1,]
    w1 = w1.squeeze().unsqueeze(-1)
    w2 = w2.squeeze().unsqueeze(-1)

    p1 = SE3(pose1.float())
    p2 = SE3(pose2.float())

    # [1,] * [6,] -> [6,]
    # [B,1] * [6,] -> [B,6] (not possible)
    # [1,] * [B,6] -> [B,6]
    # [B,1] * [B,6] -> [B,6]
    out = w1 * p1.log() + w2 * p2.log()
    out = SE3.exp(out)
    return out.data


def pose_linear_comb_vector(pose1, pose2, w1, w2):
    """
    poses in format [tx, ty, tz, qx, qy, qz, qw]
    optional batch dimension

    Args:
        pose1: tensor of shape (7,) or (B,7)
        pose2: tensor of shape (7,) or (B,7)
        w1: scalar or tensor of shape one of (6,) (B,6)
        w2: scalar or tensor of shape one of (6,) (B,6)

    Returns:
        out: tensor of shape (7,) or (B,7)
    """
    assert pose1.shape == pose2.shape, 'poses have different shapes'
    assert w1.shape == w2.shape, 'weights have different shapes'
    assert len(w1.shape) == 1 or len(w1.shape) == 2, 'w has wrong shape'
    assert w1.shape[-1] == 6, 'w has wrong shape'
    if len(w1.shape) == 2:
        assert len(pose1.shape) == 2, 'w has batch dimension but p does not'
        assert w1.shape[0] == pose1.shape[0], 'batch dimension mismatch'
    
    p1 = SE3(pose1.float())
    p2 = SE3(pose2.float())

    # [6,] * [6,] -> [6,]
    # [B,6] * [6,] -> [B,6] (not possible)
    # [6,] * [B,6] -> [B,6]
    # [B,6] * [B,6] -> [B,6]
    out = w1 * p1.log() + w2 * p2.log()
    out = SE3.exp(out)
    return out.data


def pose_linear_comb_matrix(pose1, pose2, w1, w2):
    """
    poses in format [tx, ty, tz, qx, qy, qz, qw]
    optional batch dimension

    Args:
        pose1: tensor of shape (7,) or (B,7)
        pose2: tensor of shape (7,) or (B,7)
        w1: scalar or tensor of shape one of (6,6) (B,6,6)
        w2: scalar or tensor of shape one of (6,6) (B,6,6)

    Returns:
        out: tensor of shape (7,) or (B,7)
    """
    assert pose1.shape == pose2.shape, 'poses have different shapes'
    assert w1.shape == w2.shape, 'weights have different shapes'
    assert len(w1.shape) == 2 or len(w1.shape) == 3, 'w has wrong shape'
    assert w1.shape[-2:] == (6,6), 'w has wrong shape'
    if len(w1.shape) == 3:
        assert len(pose1.shape) == 2, 'w has batch dimension but p does not'
        assert w1.shape[0] == pose1.shape[0], 'batch dimension mismatch'

    p1 = SE3(pose1.float())
    p2 = SE3(pose2.float())

    # [6,] -> [6,1]
    # [B,6] -> [B,6,1]
    p1_log = p1.log().unsqueeze(-1)
    p2_log = p2.log().unsqueeze(-1)

    # [6,6] @ [6,1] -> [6,1]
    # [B,6,6] @ [6,1] -> [B,6,1]  (not possible)
    # [6,6] @ [B,6,1] -> [B,6,1]
    # [B,6,6] @ [B,6,1] -> [B,6,1]
    out = w1 @ p1_log + w2 @ p2_log

    # [6,1] -> [6,]
    # [B,6,1] -> [B,6]
    out = out.squeeze(-1)
    out = SE3.exp(out)
    return out.data
