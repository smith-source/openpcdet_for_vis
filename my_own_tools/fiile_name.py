import os
import pickle

root_path = '/home/smith/my_projects/data/fog/fog_random'

a = os.listdir(root_path)

name_list = []

for i in a:
    name_list.append(i)
    num_index = i.split('.')[0]
    file_name = num_index + '.bin'
    os.rename(os.path.join(root_path, i), os.path.join(root_path, file_name))

with open(os.path.join(root_path, 'org_file_name.pickle'), 'wb') as f:
    pickle.dump(name_list, f)