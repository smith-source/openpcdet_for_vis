import numpy as np
from tools.visual_utils import open3d_vis_utils as V
import pickle

# # points = np.fromfile('/home/smith/my_projects/data/kitti/training/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)

points = np.fromfile('/home/smith/my_projects/data/random/003328.bin', dtype=np.float32).reshape(-1, 4)         # # confirm the np.float64,
points = points[:, :3]

# with open('/home/smith/my_projects/data/02.pickle', 'rb') as f:
#     a = pickle.load(f)
#     points = a.cpu().numpy().reshape(-1,3)



# with open('/home/smith/my_projects/SASA/train_pic/3dssd/512.pickle', 'rb') as f:
#     a = pickle.load(f)
#     points = a.cpu().numpy().reshape(-1,3)


V.draw_scenes(
                points=points,
            )
