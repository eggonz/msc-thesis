import numpy as np
import torch
import torch.nn.functional as F

from src.fast_flow_net.FastFlowNet_v2 import FastFlowNet


_DIV_FLOW = 20.0
_DIV_SIZE = 64


def _centralize(img1, img2):
    """normalize the input images"""
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def load_frozen_model(ckpt_path):
    """
    Args:
        ckpt_path: path to the checkpoint file

    Returns:
        model: FastFlowNet model
    """
    model = FastFlowNet().cuda().eval()
    model.load_state_dict(torch.load(ckpt_path))
    model = model.cuda()  # TODO add arg 'device'
    return model


def predict_flow(model, img1, img2):
    """
    Args:
        model: FastFlowNet model
        img1: [1, 3, h, w] float32
        img2: [1, 3, h, w] float32

    Returns:
        flow: [1, 2, h, w] float32
    """

    img1, img2, _ = _centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % _DIV_SIZE != 0 or width % _DIV_SIZE != 0:
        input_size = (
            int(_DIV_SIZE * np.ceil(height / _DIV_SIZE)), 
            int(_DIV_SIZE * np.ceil(width / _DIV_SIZE))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda().contiguous()  # TODO add arg 'device'

    output = model(input_t).data

    flow = _DIV_FLOW * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    return flow
