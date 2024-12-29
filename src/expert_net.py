"""
This module contains the expert network class.

The expert network is a neural network that is trained to predict the weight for each expert in the ensemble.

Some helpful definitions:
    B: batch size
    H: height of the image
    W: width of the image

The inputs to the network are the following stacked features:
- Image: [B, 3, H, W]
- Depth: [B, 1, H, W]
- Image gradients: [B, 2, H, W]
- Depth gradients: [B, 2, H, W]
- Optical flow: [B, 2, H, W]
- Normals: [B, 3, H, W]
TOTAL: [B, F, H, W] where F is the number of features

The output of the network has shape [B, 1] (logits)
The output is the weight for the expert: weight = sigmoid(logits)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.loggers import logger
from src.utils import image_ops


# positional encoding from nerf
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


# Encoders

class PoseEncoder(nn.Module):
    # from PAE

    def __init__(self, encoder_dim, apply_positional_encoding=True,
                 num_encoding_functions=6, shallow_mlp=False):

        super(PoseEncoder, self).__init__()
        self.apply_positional_encoding = apply_positional_encoding
        self.num_encoding_functions = num_encoding_functions
        self.include_input = True
        self.log_sampling = True
        x_dim = 3
        q_dim = 4
        if self.apply_positional_encoding:
            x_dim = x_dim + self.num_encoding_functions * x_dim * 2
            q_dim = q_dim + self.num_encoding_functions * q_dim * 2
        if shallow_mlp:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, encoder_dim))
        else:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,128),
                                           nn.ReLU(),
                                           nn.Linear(128,256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim))
        self.x_dim = x_dim
        self.q_dim = q_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pose):
        """
        Args:
            pose: torch.Tensor [B, 7] in format [tx, ty, tz, qx, qy, qz, qw]
        Returns:
            latent_x: torch.Tensor [B, encoder_dim] logit
            latent_q: torch.Tensor [B, encoder_dim] logit
        """
        if self.apply_positional_encoding:
            encoded_x = positional_encoding(pose[:, :3])
            encoded_q = positional_encoding(pose[:, 3:])
        else:
            encoded_x = pose[:, :3]
            encoded_q = pose[:, 3:]

        latent_x = self.x_encoder(encoded_x)
        latent_q = self.q_encoder(encoded_q)
        return latent_x, latent_q


class TimeEncoder(nn.Module):
    def __init__(self, output_dim, shallow_mlp=False):
        super(TimeEncoder, self).__init__()
        if shallow_mlp:
            self.fc = nn.Sequential(nn.Linear(1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, output_dim))
        else:
            self.fc = nn.Sequential(nn.Linear(1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, output_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, t):
        """
        Args:
            t: torch.Tensor [B, 1]
        Returns:
            latent_t: torch.Tensor [B, output_dim] logit
        """
        return self.fc(t)


class FeatureEncoder2d(nn.Module):
    _FULLRES = False  # NOTE: change with expert-resolution
    # input [320,640] -> pool4 output [20,40]
    # input [680,1200] -> pool5 output [21,37] (full-res)

    def __init__(self, input_dim, output_dim):
        super(FeatureEncoder2d, self).__init__()

        # Define the network architecture

        hidden_dims = [64, 128, 128, 256]
        if FeatureEncoder2d._FULLRES:
            hidden_dims += [256]
        fc_hidden_dim = 1024

        self.conv1 = nn.Conv2d(input_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(hidden_dims[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        if FeatureEncoder2d._FULLRES:
            self.conv5 = nn.Conv2d(hidden_dims[3], hidden_dims[4], kernel_size=3, stride=1, padding=1)
            self.relu5 = nn.ReLU()
            self.bn5 = nn.BatchNorm2d(hidden_dims[4])
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        _input_shape = (680,1200) if FeatureEncoder2d._FULLRES else (320,640)
        _num_levels = len(hidden_dims)
        for i in range(_num_levels):
            _input_shape = (_input_shape[0] // 2, _input_shape[1] // 2)
            logger.debug(f'wNet: pool{i+1} shape: {_input_shape}')

        self.fc1 = nn.Linear(hidden_dims[_num_levels-1] * _input_shape[0] * _input_shape[1], fc_hidden_dim)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)  # output logits

        # # weight initialization
        # for m in [self.fc1, self.fc2]:
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        # for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor [B, input_dim, H, W] where input_dim is the number of input channels/features
        Returns:
            logits: torch.Tensor [B, output_dim]
        """
        B = x.size(0)

        logger.debug(f'wNet-fwd: input shape: {x.shape}')

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        if FeatureEncoder2d._FULLRES:
            x = self.conv5(x)
            x = self.relu5(x)
            x = self.bn5(x)
            x = self.pool5(x)
            logger.debug(f'wNet-fwd: pool5 out shape: {x.shape}')
        else:
            logger.debug(f'wNet-fwd: pool4 out shape: {x.shape}')

        x = x.view(B, -1)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)

        logger.debug(f'wNet-fwd: output shape: {x.shape}')
        return x


# ExpertNet


class ExpertNet(nn.Module):
    MAX_TIME = 2000

    def __init__(self, input_dim=32, latent_dim=256, output_dim=1):
        super(ExpertNet, self).__init__()
        self.feat_encoder = FeatureEncoder2d(input_dim=input_dim, output_dim=latent_dim)
        self.pose_encoder = PoseEncoder(encoder_dim=latent_dim, apply_positional_encoding=True, shallow_mlp=True)
        self.time_encoder = TimeEncoder(output_dim=latent_dim, shallow_mlp=True)

        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Sequential(
            nn.Linear(4 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, init_pose, t=None):
        """
        Args:
            feats: [B, F, H, W] where F is the number of features
            init_pose: [B, 7]
            t: [B,], optional

        Returns:
            logits: [B, output_dim]
        """
        B = feats.shape[0]
        assert feats.shape[:2] == (B, 32), f'{feats.shape=}'
        assert init_pose.shape == (B, 7), f'{init_pose.shape=}'
        if t is not None:
            assert t.shape == (B,), f'{t.shape=}'
            t = torch.clamp(t, 0, ExpertNet.MAX_TIME) / ExpertNet.MAX_TIME  # normalize time
        else:
            t = torch.zeros(B, device=feats.device)
        t = t.unsqueeze(1)  # [B,] -> [B,1]

        logger.debug(f'input dims: {feats.shape=} {init_pose.shape=} {t.shape=}')

        latent_feats = self.feat_encoder(feats)  # [B, latent_dim]
        latent_x, latent_q = self.pose_encoder(init_pose)  # [B, latent_dim], [B, latent_dim]
        latent_time = self.time_encoder(t)  # [B, latent_dim]

        logger.debug(f'encoder dims: {latent_feats.shape=} {latent_x.shape=} {latent_q.shape=} {latent_time.shape=}')

        latent = torch.cat([latent_feats, latent_x, latent_q, latent_time], dim=1)  # [B, 4*latent_dim]
        logger.debug(f'fc input dims: {latent.shape=}')

        latent = F.relu(latent)
        latent = self.dropout(latent)
        logits = self.fc(latent)  # [B, output_dim]
        logger.debug(f'output dims: {logits.shape=}')
        return logits
    

class ExpertPoseNet(nn.Module):
    """
    z_feat = FeatureEncoder2d(feats)
    z_ini_x, z_ini_q = PoseEncoder(init_pose)
    z_t = TimeEncoder(t)

    z = fc([z_feat, z_ini_x, z_ini_q, z_t])

    z1_x, z1_q = PoseEncoder(f2f_delta)
    z2_x, z2_q = PoseEncoder(f2m_delta)

    x = fc([z, z1_x, z2_x])
    q = fc([z, z1_q, z2_q])

    delta_pred = [x, q]
    """
    MAX_TIME = 2000

    def __init__(self, input_dim=32, latent_dim=256):
        super(ExpertPoseNet, self).__init__()
        self.feat_encoder = FeatureEncoder2d(input_dim=input_dim, output_dim=latent_dim)
        self.pose_encoder = PoseEncoder(encoder_dim=latent_dim, apply_positional_encoding=True, shallow_mlp=True)
        self.time_encoder = TimeEncoder(output_dim=latent_dim, shallow_mlp=True)

        self.fc = nn.Sequential(
            nn.Linear(4 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(p=0.1)
        self.x_reg = nn.Sequential(
            nn.Linear(3 * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.q_reg = nn.Sequential(
            nn.Linear(3 * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, init_pose, expert1_delta, expert2_delta, t=None):
        """
        Args:
            feats: [B, F, H, W] where F is the number of features
            init_pose: [B, 7]
            expert1_delta: [B, 7], delta-pose prediction from expert 1
            expert2_delta: [B, 7], delta-pose prediction from expert 2
            t: [B,], optional

        Returns:
            pred_delta: [B, 7], delta-pose prediction from the expert network
        """
        B = feats.shape[0]
        assert feats.shape[:2] == (B, 32), f'{feats.shape=}'
        assert init_pose.shape == (B, 7), f'{init_pose.shape=}'
        assert expert1_delta.shape == (B, 7), f'{expert1_delta.shape=}'
        assert expert2_delta.shape == (B, 7), f'{expert2_delta.shape=}'
        if t is not None:
            assert t.shape == (B,), f'{t.shape=}'
            t = torch.clamp(t, 0, ExpertNet.MAX_TIME) / ExpertNet.MAX_TIME  # normalize time
        else:
            t = torch.zeros(B, device=feats.device)
        t = t.unsqueeze(1)  # [B,] -> [B,1]

        logger.debug(f'input dims: {feats.shape=} {init_pose.shape=} {t.shape=} {expert1_delta.shape=} {expert2_delta.shape=}')

        latent_feats = self.feat_encoder(feats)  # [B, latent_dim]
        latent_x_init, latent_q_init = self.pose_encoder(init_pose)  # [B, latent_dim], [B, latent_dim]
        latent_time = self.time_encoder(t)  # [B, latent_dim]

        logger.debug(f'encoder dims: {latent_feats.shape=} {latent_x_init.shape=} {latent_q_init.shape=} {latent_time.shape=}')

        latent = torch.cat([latent_feats, latent_x_init, latent_q_init, latent_time], dim=1)  # [B, 4*latent_dim]

        logger.debug(f'fc input dims: {latent.shape=}')
        latent = F.relu(latent)
        latent = self.fc(latent)  # [B, latent_dim]
        logger.debug(f'fc output dims: {latent.shape=}')

        latent_x_1, latent_q_1 = self.pose_encoder(expert1_delta)  # [B, latent_dim], [B, latent_dim]
        latent_x_2, latent_q_2 = self.pose_encoder(expert2_delta)

        logger.debug(f'latent dims: {latent.shape=} {latent_x_1.shape=} {latent_x_2.shape=} {latent_q_1.shape=} {latent_q_2.shape=}')

        latent_x = torch.cat([latent, latent_x_1, latent_x_2], dim=1)  # [B, 3*latent_dim]
        latent_q = torch.cat([latent, latent_q_1, latent_q_2], dim=1)

        logger.debug(f'regressor input dims: {latent_x.shape=} {latent_q.shape=}')
        latent_x = F.relu(latent_x)
        latent_q = F.relu(latent_q)
        latent_x = self.dropout(latent_x)
        latent_q = self.dropout(latent_q)
        x = self.x_reg(latent_x)  # [B, 3]
        q = self.q_reg(latent_q)  # [B, 4]
        logger.debug(f'output dims: {x.shape=} {q.shape=}')

        pred_delta = torch.cat((x, q), dim=1)  # [B, 7]
        return pred_delta


def prepare_feats(images, depths, images_prev, depths_prev):
    """
    Compute and stack the inputs for the expert network.
        images: [B, 3, H, W] f=3 range=[0,1]
        depths: [B, 1, H, W] f=1
        images_grads: [B, 3, 2, H, W] f=6
        depth_grads: [B, 1, 2, H, W] f=2
        images_tgrad: [B, 3, H, W] f=3
        depth_tgrad: [B, 1, H, W] f=1
        image_normals: [B, 3, 3, H, W] f=9
        depth_normals: [B, 1, 3, H, W] f=3
        image_flow: [B, 2, H, W] f=2
        depth_flow: [B, 2, H, W] f=2

    Args:
        images: [B, 3, H, W]
        depths: [B, 1, H, W]

    Returns:
        stacked_feats: [B, F=32, H, W] where F is the number of features
    """
    b, _, h, w = images.shape
    i_grads = image_ops.compute_spatial_gradients(images)  # [B, 3, 2, H, W]
    d_grads = image_ops.compute_spatial_gradients(depths)  # [B, 1, 2, H, W]
    i_tgrad = image_ops.compute_time_gradient(images_prev, images)  # [B, 3, H, W]
    d_tgrad = image_ops.compute_time_gradient(depths_prev, depths)  # [B, 1, H, W]
    i_normals = image_ops.compute_normals(i_grads)  # [B, 3, 3, H, W]
    d_normals = image_ops.compute_normals(d_grads)  # [B, 1, 3, H, W]
    i_flow = image_ops.compute_flow(images_prev, images)  # [B, 2, H, W]
    d_flow = image_ops.compute_flow(depths_prev, depths)  # [B, 2, H, W]
    
    stacked_feats = torch.cat([feat.view(b, -1, h, w) for feat in [
        images, depths, i_grads, d_grads, i_tgrad, d_tgrad, i_normals, d_normals, i_flow, d_flow]], dim=1)
    return stacked_feats
