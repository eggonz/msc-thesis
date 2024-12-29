import sys
from scipy.interpolate import interp1d
from skimage import filters
from skimage.color import rgb2gray
import subprocess
import os
import random
import argparse
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
sys.path.append('.')
from src import config
from src.common import update_cam
from src.utils.eval_traj import align_kf_traj
from src.utils.datasets import get_dataset
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_neural_point_cloud(slam, ckpt, device):

    slam.npc._cloud_pos = ckpt['cloud_pos']
    slam.npc._input_pos = ckpt['input_pos']
    slam.npc._input_rgb = ckpt['input_rgb']
    slam.npc._pts_num = len(ckpt['cloud_pos'])
    slam.npc.geo_feats = ckpt['geo_feats'].to(device)
    slam.npc.col_feats = ckpt['col_feats'].to(device)

    cloud_pos = torch.tensor(ckpt['cloud_pos'], device=device)
    slam.npc.index_train(cloud_pos)
    slam.npc.index.add(cloud_pos)

    print(
        f'Successfully loaded neural point cloud, {slam.npc.index.ntotal} points in total.')


def load_ckpt(cfg, slam):
    """
    Saves mesh of already reconstructed model from checkpoint file. Makes it 
    possible to remesh reconstructions with different settings and to draw the cameras
    """

    assert cfg['mapping']['save_selected_keyframes_info'], 'Please save keyframes info to help run this code.'

    ckptsdir = f'{slam.output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('\nGet ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
        else:
            raise ValueError(f'Check point directory {ckptsdir} is empty.')
    else:
        raise ValueError(f'Check point directory {ckptsdir} not found.')

    return ckpt


class DepthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('depth_')])
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('color_')])

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = int(base[-5:])
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = np.load(self.depth_files[idx])
        image = np.load(self.image_files[idx])

        if self.transform:
            depth = self.transform(depth)
            image = self.transform(image)

        return depth, image


def generate_mesh_kf(config_path,rendered_path="rendered_every_keyframe",mesh_name_suffix="kf"):
    cfg = config.load_config(config_path, "configs/expert_slam.yaml")
    # define variables for dynamic query radius computation
    output = cfg['data']['output']
    device = cfg['mapping']['device']
    offline_video = f"{output}/{cfg['offline_video']}"
    dataset = DepthImageDataset(root_dir=f'{output}/{rendered_path}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    warmup = cfg['tracking']['warmup']
    scene_name = cfg["scene"]
    mesh_name = f'{scene_name}_{mesh_name_suffix}.ply'
    mesh_out_file = f'{output}/mesh/{mesh_name}'

    H, W, fx, fy, cx, cy = update_cam(cfg)
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    print('\nStarting to integrate the mesh...')
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)

    os.makedirs(f'{output}/mesh/mid_mesh', exist_ok=True)
    frame_reader = get_dataset(cfg, device=device)
    _,_,scene_scale,traj_est,traj_ref = align_kf_traj(offline_video,frame_reader,return_full_est_traj=True)
    if cfg['tracking']['gt_camera']:
        traj = traj_ref
        scene_scale = 1.0
    else:
        traj = traj_est
    video = np.load(offline_video)

    v_idx_offset = 0
    for i, (depth, color) in tqdm(enumerate(dataloader),total=len(dataset)):
        index = dataset.indices[i]
        video_idx = i + v_idx_offset
        while index!=int(video['timestamps'][video_idx]):
            assert(int(video['timestamps'][video_idx])<index)
            print(f"Skip frame {int(video['timestamps'][video_idx])} (v_idx:{video_idx}) because rendered image and depth are not found.")
            v_idx_offset += 1
            video_idx = i + v_idx_offset
            
        depth = depth[0].cpu().numpy() * scene_scale
        color = color[0].cpu().numpy()
        
        c2w = traj.poses_se3[video_idx]
        w2c = np.linalg.inv(c2w)

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

        if i > 0 and cfg["meshing"]["mesh_freq"] > 0 and (i % cfg["meshing"]["mesh_freq"]) == 0:
            o3d_mesh = volume.extract_triangle_mesh()
            o3d_mesh = o3d_mesh.translate(compensate_vector)
            o3d.io.write_triangle_mesh(
                f"{output}/mesh/mid_mesh/frame_{cfg['mapping']['every_frame']*i}_mesh.ply", o3d_mesh)
            print(
                f"saved intermediate mesh until frame {cfg['mapping']['every_frame']*i}.")
    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{output}/mesh',
            f'vertices_pos_{mesh_name_suffix}.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    print(f"Final mesh file is saved: {mesh_out_file}")
    print('🕹️ Meshing finished.')


def generate_mesh_every(config_path, traj_file="full_traj_aligned.npy",rendered_path="rendered_every_frame",mesh_name_suffix="every", timestamps=None):
    cfg = config.load_config(config_path, "configs/expert_slam.yaml")
    # define variables for dynamic query radius computation
    output = cfg['data']['output']
    device = cfg['mapping']['device']
    offline_video = f"{output}/{cfg['offline_video']}"
    traj_path = f"{output}/traj/{traj_file}"  # NOTE: ExpertSLAM uses out_traj, not full
    dataset = DepthImageDataset(root_dir=f'{output}/{rendered_path}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    scene_name = cfg["scene"]
    mesh_name = f'{scene_name}_{mesh_name_suffix}.ply'
    mesh_out_file = f'{output}/mesh/{mesh_name}'

    H, W, fx, fy, cx, cy = update_cam(cfg)
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    print('\nStarting to integrate the mesh...')
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)
    if cfg["meshing"]["mesh_freq"] > 0:
        os.makedirs(f'{output}/mesh/mid_mesh_every', exist_ok=True)
    video = np.load(offline_video)
    traj = np.load(traj_path)
    scene_scale = video["scale"]
    if timestamps is not None:
        timestamps = np.array(timestamps, dtype=int)  # [N,]

    for i, (depth, color) in tqdm(enumerate(dataloader),total=len(dataset)):
        index = dataset.indices[i]            
        depth = depth[0].cpu().numpy() * scene_scale
        color = color[0].cpu().numpy()
        
        if timestamps is not None:
            if int(index) not in timestamps:
                continue
            c2w = traj[timestamps==int(index)][0]
        else:
            c2w = traj[index]
        w2c = np.linalg.inv(c2w)

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

        if i > 0 and cfg["meshing"]["mesh_freq"] > 0 and (i % cfg["meshing"]["mesh_freq"]) == 0:
            o3d_mesh = volume.extract_triangle_mesh()
            o3d_mesh = o3d_mesh.translate(compensate_vector)
            o3d.io.write_triangle_mesh(
                f"{output}/mesh/mid_mesh/frame_{cfg['mapping']['every_frame']*i}_mesh.ply", o3d_mesh)
            print(
                f"saved intermediate mesh until frame {cfg['mapping']['every_frame']*i}.")
    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{output}/mesh',
            f'vertices_pos_{mesh_name_suffix}.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    print(f"Final mesh file is saved: {mesh_out_file}")
    print('🕹️ Meshing finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configs for Mono-Point-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    args = parser.parse_args()
    generate_mesh_kf(args.config)
