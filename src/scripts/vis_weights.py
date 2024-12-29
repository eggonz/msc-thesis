import numpy as np
from PIL import Image

folder = "/scratch_net/rollgardina/zhangga/euler_work/src/expert_slam/output/Replica/room0/"
graph_path = folder + "factor_graph.npz"
graph = np.load(graph_path)

weight = graph['weight']
weight_up = graph['weight_up']

weight_eg = weight[0,0,:,:,0]
weight_up_eg = weight_up[0,0,:,:,0]


Image.fromarray((weight_eg * 6553.5).astype(np.int32)).save(folder+"weigth.png")
Image.fromarray((weight_up_eg * 6553.5).astype(np.int32)).save(folder+"weigth_up.png")
