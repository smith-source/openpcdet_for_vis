import pickle
import json
import os


path = ''
txt_path = ''
with open(path, 'rb') as f:
    a = pickle.load(f)


for i in a:

    index = i['point_cloud']['lidar_idx']
    file_path = os.path.join(txt_path, index)
