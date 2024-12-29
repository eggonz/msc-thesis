import cv2
import torch
import lietorch

from collections import OrderedDict
from src.droid_net import DroidNet

import src.geom.projective_ops as pops
from src.modules.corr import CorrBlock
from src.mono_estimators import get_mono_estimator,predit_mono_depth,predit_mono_normal
from src.utils.datasets import load_mono_depth

class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        if cfg["mono_prior"]["predict_online"]:
            self.mono_depth_estimator = get_mono_estimator(cfg,"depth")

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, gt_depth, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        # inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = image[None, :, :].to(self.device).clone()
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        if (self.cfg["mono_prior"]["predict_online"]) \
            and (tstamp % self.cfg['mapping']['every_frame'] == 0):
            predit_mono_depth(self.mono_depth_estimator,tstamp,image,self.cfg,self.device)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            if self.cfg["mono_prior"]["predict_online"]:
                # mono_depth = predit_mono_depth(self.mono_depth_estimator,tstamp,image,self.cfg,self.device)
                mono_depth = gt_depth.clone()
            else:
                # mono_depth = load_mono_depth(tstamp,self.cfg)
                mono_depth = gt_depth.clone()
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])
        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                if self.cfg["mono_prior"]["predict_online"]:
                    # mono_depth = predit_mono_depth(self.mono_depth_estimator,tstamp,image,self.cfg,self.device)
                    mono_depth = gt_depth
                else:
                    # mono_depth = load_mono_depth(tstamp,self.cfg)
                    mono_depth = gt_depth
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / 8.0, gmap, net[0], inp[0])

            else:
                self.count += 1
