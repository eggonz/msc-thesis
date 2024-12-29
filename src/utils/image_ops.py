import torch
import torch.nn as nn
from kornia.filters import SpatialGradient

from src.fast_flow_net.flow import load_frozen_model, predict_flow


def compute_spatial_gradients(images):
    """
    Args:
        images: images or depths, (B, C, H, W) tensor

    Returns:
        gradients: spatial gradient vector, (B, C, 2, H, W) tensor
    """
    return SpatialGradient()(images)


def compute_time_gradient(images1, images2):
    """
    Args:
        images1: images or depths at first timestamp, (B, C, H, W) tensor
        images2: images or depths at second timestamp, (B, C, H, W) tensor

    Returns:
        gradients: time gradient vector, (B, C, H, W) tensor
    """
    return images2 - images1


def compute_normals(grads):
    """
    Args:
        grads: image or depth gradients, (B, C, 2[xy], H, W) tensor

    Returns:
        normals: unitary vector normals, (B, C, 3[xyz], H, W) tensor
    """
    ones = torch.ones_like(grads[:, :, :1, :, :])  # (B, C, 1, H, W)
    # n = [-gx, -gy, 1]
    n = torch.cat([-grads, ones], dim=2)  # (B, C, 3[xyz], H, W)
    n_norm = torch.norm(n, dim=2, keepdim=True)  # (B, C, 1, H, W)
    n_unit = n / n_norm  # (B, C, 3[xyz], H, W)
    return n_unit


flow_net = load_frozen_model('pretrained/fastflownet_ft_mix.pth')


def compute_flow(images1, images2):
    """
    Args:
        images1: images or depths at first timestamp, (B, C, H, W) tensor
        images2: images or depths at second timestamp, (B, C, H, W) tensor

    Returns:
        flow: (B, 2, H, W) tensor
    """
    ch = images1.shape[1]  # 3 for images, 1 for depths
    if ch == 1:
        # input must be 3-channel images
        images1 = images1.expand(-1, 3, -1, -1).clone()
        images2 = images2.expand(-1, 3, -1, -1).clone()
    return predict_flow(flow_net, images1, images2)
