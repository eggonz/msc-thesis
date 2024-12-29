from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PosePath3D
import numpy as np
from src.utils.datasets import get_dataset
from src import config

cfg = config.load_config(
        './configs/Replica/room0_mono.yaml', './configs/expert_slam.yaml')
output = f"{cfg['data']['output']}/{cfg['setting']}"
dataset = get_dataset(cfg, None, device='cuda:0')

graph_path = f"{output}/factor_graph.npz"
offline_graph = np.load(graph_path)
est_poses = offline_graph['poses']
timestamps = offline_graph['timestamps']

before = graph_path = f"{output}/before_filling_video.npz"
before_video = np.load(before)
before_poses = before_video['poses']

after = graph_path = f"{output}/after_filling_video.npz"
after_video = np.load(after)
after_poses = after_video['poses']



traj_ref = []
traj_est = []

for i in range(timestamps.shape[0]):
    idx = int(timestamps[i])
    traj_est.append(est_poses[i])
    traj_ref.append(dataset.poses[idx])

traj_est = PosePath3D(poses_se3=traj_est)
traj_ref = PosePath3D(poses_se3=traj_ref)


result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                                      pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

print(result.pretty_str())
