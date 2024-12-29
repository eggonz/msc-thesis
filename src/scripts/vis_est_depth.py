import numpy as np
import PIL
from PIL import Image

video_path = "/cluster/work/cvl/esandstroem/src/expert_slam/output/TUM_RGBD/freiburg1_room_filtered_depth/video_before.npz"
video = np.load(video_path)
save_prefix = "/cluster/work/cvl/esandstroem/src/expert_slam/output/TUM_RGBD/freiburg1_room_filtered_depth/est_depth"

# for i in range(len(video["depths"])):
for i in range(20):
    d = video["depths"][i]
    mask = video["valid_depth_masks"][i]
    d[~mask] = 0
    t = int(video["timestamps"][i])
    Image.fromarray((d*5000.0).astype(np.int32)).save(f"{save_prefix}/{t}.png")