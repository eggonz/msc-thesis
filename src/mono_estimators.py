import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

def get_mono_estimator(cfg,mode):
    '''
    cfg: config
    mode: (str) "depth" | "normal"
    '''
    device = cfg["mapping"]["device"]
    if mode == "depth":
        depth_model = cfg["mono_prior"]["depth"]
        depth_pretrained = cfg["mono_prior"]["depth_pretrained"]
        if depth_model == "omnidata":
            model = get_omnidata_model(depth_pretrained, device, 1)
    elif mode == "normal":    
        normal_model = cfg["mono_prior"]["normal"]
        normal_pretrained = cfg["mono_prior"]["normal_pretrained"]
        if normal_model == "omnidata":
            model = get_omnidata_model(normal_pretrained, device, 3)    
    return model

def get_omnidata_model(pretrained_path, device, num_channels):
    from src.mono_priors.omnidata.modules.midas.dpt_depth import DPTDepthModel
    model = DPTDepthModel(backbone='vitb_rn50_384',num_channels=num_channels)
    checkpoint = torch.load(pretrained_path)
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)

    return model.to(device).eval()

@torch.no_grad()
def predit_mono_depth(model,idx,input,cfg,device):
    '''
    input: tensor (3,H,W)
    '''
    depth_model = cfg["mono_prior"]["depth"]
    output_dir = cfg['data']['output']
    if depth_model == "omnidata":
        s = cfg["cam"]["H_out"]
        # image_size = (s,s)
        image_size = (512,512)
        input_size = input.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                            transforms.Normalize(mean=0.5, std=0.5)])
        img_tensor = trans_totensor(input).to(device)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), input_size, mode='bicubic').squeeze(0)
        output = output.clamp(0,1).squeeze() #[H,W]
    else:
        raise NotImplementedError
    
    output_path_np = f"{output_dir}/mono_priors/depths/{idx:05d}.npy"
    final_depth = output.detach().cpu().float().numpy()
    np.save(output_path_np, final_depth)

    return output
    # output_path_img = f"{output_dir}/mono_priors/depths/{idx:05d}.png"
    # Image.fromarray((final_depth *150* 6553.5).astype(np.int32)).save(output_path_img)
    # exit()

@torch.no_grad()
def predit_mono_normal(model,idx,input,cfg,device,save_img=False):
    normal_model = cfg["mono_prior"]["normal"]
    output_dir = cfg['data']['output']
    if normal_model == "omnidata":
        s = cfg["cam"]["H_out"]
        image_size = (s,s)
        input_size = input.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size)])
        img_tensor = trans_totensor(input).to(device)        
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output, input_size, mode='nearest-exact').squeeze(0)
        #[3,H,W]
    else:
        raise NotImplementedError
    
    output_path_np = f"{output_dir}/mono_priors/normals/{idx:05d}.npy"
    final_depth = output.detach().cpu().float().numpy()
    np.save(output_path_np, final_depth)

    if save_img:
        trans_topil = transforms.ToPILImage()
        output_path_png = f"{output_dir}/mono_priors/normals/{idx:05d}.png"
        trans_topil(output).save(output_path_png)

    return output