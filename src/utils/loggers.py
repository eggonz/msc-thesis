import os
import logging

import torch
import wandb
from torch.multiprocessing import Value
import pandas as pd

LOG_FORMAT = '[%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


logger = logging.getLogger('ExpertSLAM')
logger.setLevel(logging.INFO)


class WandbLogger:
    def __init__(self, cfg):
        self._step = Value('i', 0)
        self._logger = wandb.init(
            resume="allow",
            config=cfg,
            project=cfg["setting"],
            group=cfg["dataset"],
            name=cfg["expname"],
            settings=wandb.Settings(code_dir="."),
            dir=cfg["wandb_folder"],
            tags=[cfg["scene"]])
        self._logger.log_code(".")

    def set_step(self, step):
        self._step.value = step

    def log(self, *args, **kwargs):
        # use current step if not provided, don't increment unless explicitly called
        self._logger.log(*args, **kwargs, step=self._step.value)

    def finish(self):
        self._logger.finish()


class TrainerLogger:
    def __init__(self):
        self.data = []
        self._scene = None
        self._rep = None
        self._fold = None
        self._epoch = None
        self._batch = None

    def update(self, **kwargs):
        self._scene = kwargs.get("scene", self._scene)
        self._rep = kwargs.get("rep", self._rep)
        self._fold = kwargs.get("fold", self._fold)
        self._epoch = kwargs.get("epoch", self._epoch)
        self._batch = kwargs.get("batch", self._batch)

    def log(self, **kwargs):
        self.update(**kwargs)
        self.data.append({
            "scene": self._scene,
            "rep": self._rep,
            "fold": self._fold,
            "epoch": self._epoch,
            "batch": self._batch,
            **kwargs
        })

    def save_csv(self, path):
        df = pd.DataFrame(self.data)
        # cannot convert to int because of NaN values
        df.to_csv(path, index=False)


class CkptLogger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, mapper):
        self.verbose = mapper.verbose
        self.ckptsdir = mapper.ckptsdir
        # self.gt_c2w_list = mapper.gt_c2w_list  # FIXME expert does not have this
        # self.estimate_c2w_list = mapper.estimate_c2w_list  # FIXME expert does not have this
        self.decoders = mapper.decoders

    def log(self, idx, keyframe_dict, keyframe_list, npc, exposure_feat=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'geo_feats': npc.get_geo_feats().cpu(),
            'col_feats': npc.get_col_feats().cpu(),
            'cloud_pos': npc.cloud_pos().cpu(),
            'pts_num': npc.pts_num(),
            'input_pos': npc.input_pos().cpu(),
            'input_rgb': npc.input_rgb().cpu(),

            'decoder_state_dict': self.decoders.state_dict(),
            # 'gt_c2w_list': self.gt_c2w_list,
            # 'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'keyframe_dict': keyframe_dict,
            # 'selected_keyframes': selected_keyframes,
            'idx': idx,
            "exposure_feat_all": torch.stack(exposure_feat, dim=0)
            if exposure_feat is not None
            else None,
        }, path)

        if self.verbose:
            print('Saved checkpoints at', path)
