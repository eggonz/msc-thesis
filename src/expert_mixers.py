
import abc

import torch
import lietorch
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils import pose_ops
from src.utils.loggers import logger
from src.expert_net import ExpertNet, ExpertPoseNet, prepare_feats


def get_fixed_mixer(mix_method):
    """ for fixed and oracle mixers """
    assert mix_method != 'psnr', 'use psnr_expert.PsnrExpert instead'
    assert not mix_method.startswith('learned'), 'use expert_net.LearnedMixer instead'
    # Fixed
    if mix_method == 'mean':
        return MixerMean()
    elif mix_method == 'f2f':
        return MixerF2f()
    elif mix_method == 'f2m':
        return MixerF2m()
    # Oracle
    elif mix_method == 'gt':
        return MixerGt()
    elif mix_method == 'best':
        return MixerBest()
    else: 
        raise NotImplementedError(f'mix_method={mix_method} not implemented')


def get_learned_mixer_class(mix_method):
    """ for learned mixers """
    assert mix_method.startswith('learned')
    classes = (MixerSlerp, MixerLinear, MixerMatrix, MixerLie, MixerPose)
    for cls in classes:
        if cls.mix_method == mix_method:
            return cls
    else:
        raise not NotImplementedError(f"Unknown mix_method: {mix_method}")


def load_learned_mixer_ckpt(path, device):
    state_dict = torch.load(path)
    mix_method = state_dict['mix_method']
    mixer_cls = get_learned_mixer_class(mix_method)
    mixer = mixer_cls(device)
    mixer.load_state_dict(state_dict)
    return mixer


# Fixed Mixers

class FixedMixer(abc.ABC):
    is_oracle = False

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, f2f_delta, f2m_delta):
        pass

class MixerMean(FixedMixer):
    def __call__(self, f2f_delta, f2m_delta):
        if f2f_delta is not None and f2m_delta is not None:
            return pose_ops.pose_slerp(f2f_delta, f2m_delta, 0.5)
        elif f2f_delta is not None:
            return f2f_delta.clone()
        elif f2m_delta is not None:
            return f2m_delta.clone()
        return None
    
class MixerF2f(FixedMixer):
    def __call__(self, f2f_delta, f2m_delta):
        if f2f_delta is not None:
            return f2f_delta.clone()
        return None

class MixerF2m(FixedMixer):
    def __call__(self, f2f_delta, f2m_delta):
        if f2m_delta is not None:
            return f2m_delta.clone()
        return None


# Oracle Mixers

class OracleMixer(abc.ABC):
    is_oracle = True

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, f2f_delta, f2m_delta, gt_delta):
        pass

class MixerGt(OracleMixer):
    def __call__(self, f2f_delta, f2m_delta, gt_delta):
        return gt_delta.clone()
    
class MixerBest(OracleMixer):
    def __call__(self, f2f_delta, f2m_delta, gt_delta):
        if f2f_delta is not None and f2m_delta is not None:
            loss_f2f = pose_ops.pose_loss(f2f_delta, gt_delta)
            loss_f2m = pose_ops.pose_loss(f2m_delta, gt_delta)
            if loss_f2f < loss_f2m:
                return f2f_delta.clone()
            return f2m_delta.clone()
        elif f2f_delta is not None:
            return f2f_delta.clone()
        elif f2m_delta is not None:
            return f2m_delta.clone()
        return None


# Learned Mixer (base)

class LearnedMixer(abc.ABC):
    def __init__(self, device):
        self.device = device
        self.net = ...  # type: ExpertNet

    @abc.abstractmethod
    def _mix(self, f2f_delta, f2m_delta, pred):
        """
        Args:
            f2f_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw]
            f2m_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw]
            pred: torch.Tensor, output of self.net forward [B,...]

        Returns:
            mix_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw]
        """
        pass

    def _prepare_input(self, t, image, depth, init_pose, f2f_delta, f2m_delta, image_prev, depth_prev):
        """
        Args:
            t: torch.Tensor [B,]
            image: torch.Tensor [B,3,H,W]
            depth: torch.Tensor [B,1,H,W]
            init_pose: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], absolute pose
            f2f_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            f2m_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            image_prev: torch.Tensor [B,3,H,W]
            depth_prev: torch.Tensor [B,1,H,W]

        Returns:
            *inputs: torch.Tensors [B,...]
        """
        feats = prepare_feats(image, depth, image_prev, depth_prev)
        t = None  # TODO ignoring time for now!
        return feats, init_pose, t


    def __call__(self, t, image, depth, init_pose, f2f_delta, f2m_delta, image_prev, depth_prev, single_sample_raw=False, return_log_info=False):
        """
        Args (single_sample_raw=False):
            t: torch.tensor [B,]
            image: torch.Tensor [B,3,H,W]
            depth: torch.Tensor [B,1,H,W]
            init_pose: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], absolute pose
            f2f_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            f2m_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            prev_image: torch.Tensor [B,3,H,W]
            prev_depth: torch.Tensor [B,1,H,W]

        Args (single_sample_raw=True):
            t: int
            image: torch.Tensor [H,W,3]
            depth: torch.Tensor [H,W]
            init_pose: torch.Tensor [7] in format [tx, ty, tz, qx, qy, qz, qw], absolute pose 
            f2f_delta: torch.Tensor [7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            f2m_delta: torch.Tensor [7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
            prev_image: torch.Tensor [H,W,3]
            prev_depth: torch.Tensor [H,W]

        Returns (single_sample_raw=False):
            mix_delta: torch.Tensor [B,7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose

        Returns (single_sample_raw=True):
            mix_delta: torch.Tensor [7] in format [tx, ty, tz, qx, qy, qz, qw], relative to init_pose
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        assert (t!=0).all() and image_prev is not None and depth_prev is not None, f'need 2 timestamps'
        assert f2f_delta is not None and f2m_delta is not None, f'need poses to mix'

        if single_sample_raw:
            # make single sample batch
            t = torch.tensor([t], device=self.device) # int -> [1,]
            image = image.unsqueeze(0).permute(0, 3, 1, 2)  # [H,W,3] -> [1,3,H,W]
            depth = depth.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            image_prev = image_prev.unsqueeze(0).permute(0, 3, 1, 2)  # [H,W,3] -> [1,3,H,W]
            depth_prev = depth_prev.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            init_pose = init_pose.unsqueeze(0)  # [7] -> [1,7]
            f2f_delta = f2f_delta.unsqueeze(0)  # [7] -> [1,7]
            f2m_delta = f2m_delta.unsqueeze(0)  # [7] -> [1,7]

        # prepare input
        inp = self._prepare_input(t, image, depth, init_pose, f2f_delta, f2m_delta, image_prev, depth_prev)
        inp = (i.to(self.device) if i is not None else None for i in inp)

        # forward pass
        pred = self.net(*inp)

        # mix
        mix_delta, log_info = self._mix(f2f_delta.to(self.device), f2m_delta.to(self.device), pred)

        logger.debug(f'mixer log info: {log_info}')
        assert mix_delta.shape[1:] == (7,), f'{mix_delta.shape=}'

        if single_sample_raw:
            mix_delta = mix_delta[0]
        if return_log_info:
            return mix_delta, log_info
        return mix_delta
    
    def clone(self):
        mixer = self.__class__(self.device)
        mixer.net.load_state_dict(self.net.state_dict())
        return mixer
    
    def train(self):
        self.net.train()
        self.net.requires_grad_(True)
    
    def eval(self):
        self.net.eval()
        self.net.requires_grad_(False)

    def parameters(self):
        return self.net.parameters()
    
    def save(self, path):
        state_dict = {
            'mix_method': self.mix_method,
            'net': self.net.state_dict(),
        }
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        assert self.mix_method == state_dict['mix_method'], f'{self.mix_method=} != {state_dict["mix_method"]=}'
        self.load_state_dict(state_dict)
    
    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])


# Learned Mixers

class MixerSlerp(LearnedMixer):
    """
    pred = ExpertNet(x)
    w = sigmoid(pred)
    y = slerp(y1, y2, w)
    """
    mix_method = 'learned_slerp'

    def __init__(self, device):
        super().__init__(device)
        self.net = ExpertNet(input_dim=32, latent_dim=256, output_dim=1)
        self.net.to(device)

    def _mix(self, f2f_delta, f2m_delta, pred):
        w = torch.sigmoid(pred)
        mix_delta = pose_ops.pose_slerp(f2f_delta, f2m_delta, w)
        log_info = {'w': w}
        return mix_delta, log_info


class MixerLinear(LearnedMixer):
    """
    pred = ExpertNet(x)
    w1, w2 = pred.split()
    y = 1/2 * (w1 * y1 + w2 * y2)
    """
    mix_method = 'learned_linear'

    def __init__(self, device):
        super().__init__(device)
        self.net = ExpertNet(input_dim=32, latent_dim=256, output_dim=2)
        self.net.to(device)

    def _mix(self, f2f_delta, f2m_delta, pred):
        w_f2f, w_f2m = pred[:, 0], pred[:, 1]

        # OPTION 1 (best)
        mean = pose_ops.pose_slerp(f2f_delta, f2m_delta, 0.5)
        d_f2f = pose_ops.pose_sub(f2f_delta, mean)
        d_f2m = pose_ops.pose_sub(f2m_delta, mean)
        d_mix = pose_ops.pose_linear_comb_scalar(d_f2f, d_f2m, w_f2f, w_f2m)
        mix_delta = pose_ops.pose_sum(d_mix, mean)

        # OPTION 2
        # mix_delta = pose_ops.pose_linear_comb_scalar(f2f_delta, f2m_delta, 0.5 * (1 + w_f2f), 0.5 * (1 + w_f2m))
        
        log_info = {'w_f2f': w_f2f, 'w_f2m': w_f2m}
        return mix_delta, log_info


class MixerMatrix(LearnedMixer):
    """
    pred = ExpertNet(x)
    W1, W2 = pred.split()
    y = W1 @ y1 + W2 @ y2
    """
    mix_method = 'learned_matrix'

    def __init__(self, device):
        super().__init__(device)
        self.net = ExpertNet(input_dim=32, latent_dim=256, output_dim=2*6*6)
        self.net.to(device)

    def _mix(self, f2f_delta, f2m_delta, pred):
        W = pred.view(2, 6, 6)  # [2,6,6]
        W1, W2 = W.unbind(dim=0)  # [6,6], [6,6]
        eye = torch.eye(6).to(self.device)  # [6,6]
        W_f2f = eye + W1  # [6,6]
        W_f2m = eye + W2  # [6,6]
        mix_delta = pose_ops.pose_linear_comb_matrix(f2f_delta, f2m_delta, 0.5 * W_f2f, 0.5 * W_f2m)
        log_info = {'W_f2f.det': torch.det(W_f2f), 'W_f2m.det': torch.det(W_f2m)}
        self.save_plot(W_f2f, W_f2m)
        return mix_delta, log_info

    def save_plot(self, f2f, f2m):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), width_ratios=[1, 1, 0.1])
        sns.heatmap(f2f.detach().cpu().numpy(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax1, cbar=False, square=True)
        ax1.set_title('f2f')
        ax1.axis('off')
        sns.heatmap(f2m.detach().cpu().numpy(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2, cbar=False, square=True)
        ax2.set_title('f2m')
        ax2.axis('off')
        #colorbar
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, cax=ax3)
        cbar.set_ticks([-1, 0, 1])
        plt.tight_layout()
        plt.savefig('mixer_matrix.png')  # TODO add arg for custom path
        plt.close()


class MixerLie(LearnedMixer):
    """
    pred = ExpertNet(x)  # <pred> = 0
    t1, t2 = pred.split()  # <t> = 0
    g1, g2 = lie.exp(t1), lie.exp(t2)  # <g> = I

    mix_delta = mean(g1 * f2f_delta, g2 * f2m_delta)  # <mix> = (f2f + f2m) / 2
    """
    mix_method = 'learned_lie'

    def __init__(self, device, scale=0.1):
        super().__init__(device)
        self.net = ExpertNet(input_dim=32, latent_dim=256, output_dim=2*6)
        self.net.to(device)
        self.scale = scale  # avoid large transformations

    def _mix(self, f2f_delta, f2m_delta, pred):
        # vectors in tangent space (<t> = 0)
        t1, t2 = pred.split(6, dim=1)  # [B,6], [B,6]
        # lie algebra -> lie group (<g> = I)
        g1 = lietorch.SE3.exp(t1 * self.scale)
        g2 = lietorch.SE3.exp(t2 * self.scale)
        f2f_delta = g1 * lietorch.SE3(f2f_delta)
        f2m_delta = g2 * lietorch.SE3(f2m_delta)
        mix_delta = pose_ops.pose_slerp(f2f_delta.data, f2m_delta.data, 0.5)
        log_info = {
            't1': t1.detach().cpu().numpy(),
            't2': t2.detach().cpu().numpy(),
            'g1': g1.data.detach().cpu().numpy(),
            'g2': g2.data.detach().cpu().numpy(),
            't1.norm': t1.norm(dim=1).mean().item(),
            't2.norm': t2.norm(dim=1).mean().item(),
            'g1.t.norm': g1.data[:,:3].norm(dim=1).mean().item(),
            'g2.t.norm': g2.data[:,:3].norm(dim=1).mean().item(),
            'g1.r.norm': lietorch.SO3(g1.data[:,3:]).log().norm(dim=1).mean().item(),
            'g2.r.norm': lietorch.SO3(g2.data[:,3:]).log().norm(dim=1).mean().item(),
        }
        return mix_delta, log_info


class MixerPose(LearnedMixer):
    """
    feats = prepare_feats(image, depth, image_prev, depth_prev)
    delta_pred = ExpertNetPose(feats, init_pose, t, f2f_delta, f2m_delta)
    mix_delta = delta_pred
    """
    mix_method = 'learned_pose'

    def __init__(self, device):
        super().__init__(device)
        self.net = ExpertPoseNet(input_dim=32, latent_dim=256)
        self.net.to(device)

    def _prepare_input(self, t, image, depth, init_pose, f2f_delta, f2m_delta, image_prev, depth_prev):
        feats = prepare_feats(image, depth, image_prev, depth_prev)
        t = None  # TODO ignoring time for now!
        return feats, init_pose, f2f_delta, f2m_delta, t

    def _mix(self, f2f_delta, f2m_delta, pred):
        return pred, {}
