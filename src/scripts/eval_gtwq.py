
import cv2
import numpy as np

import os
tmpdir = os.environ.get('TMPDIR')
gt_path = f'{tmpdir}/room0/results/depth000000.png'
mono_path = f'{tmpdir}/room0/mono_depth_pred/depth000000.png'
scale_shift_path = f'{tmpdir}/room0/mono_depth_pred/scale_shift.csv'

# TUM
# gt_path = '/workdir/freiburg1_desk/depth/1305031453.374112.png'
# mono_path = '/workdir/freiburg1_desk/mono_depth_pred_factor1/1305031453.323682.png'
# scale_shift_path = '/workdir/freiburg1_desk/mono_depth_pred_factor1/scale_shift.csv'

# REPLICA
# gt_path = '/workdir/room0/results/depth000000.png'
# mono_path = '/workdir/room0/mono_depth_pred_factor120/depth000000.png'
# scale_shift_path = '/workdir/room0/mono_depth_pred_factor120/scale_shift.csv'

#TUM
# gt_depth = cv2.imread(gt_path, -1).astype(np.float32)/5000
# mono_depth = cv2.imread(mono_path, -1).astype(np.float32)/5000

# REPLICA
gt_depth = cv2.imread(gt_path, -1).astype(np.float32)/6553.5
mono_depth = cv2.imread(mono_path, -1).astype(np.float32)/6553.5
scale_shift = np.loadtxt(scale_shift_path, delimiter=',')
mono_depth = scale_shift[0, 0] * mono_depth + scale_shift[0, 1]

error = np.abs(gt_depth - mono_depth)
error = error[gt_depth > 0]
print("Average Error per pixel in meters: ", np.mean(error))