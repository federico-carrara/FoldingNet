import json
import h5py
import os
from tqdm import tqdm

'''
Get a dictionary that associates ShapeNetCore labels to int.
'''

path_to_data = '/nas/groups/iber/Users/Federico_Carrara/FoldingNet_project/data/ShapeNet/ShapeNetCore_pointclouds_v2/'

file_names = os.listdir(path_to_data)
labels = set()
for file_name in file_names:

    if '.h5' not in file_name:
        continue

    # Open the HDF5 file in read mode to read the existing dataset
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as file:
        curr_labels = file['categories'][:]

    labels.update(list(curr_labels.squeeze(1)))

# Create and save dictionary
label2id ={label.decode(): i+1 for i, label in enumerate(labels)}

assert os.path.exists('../../misc/'), 'You have to execute this script from its folder.'
with open('../../misc/shapenetcoreccategory2id.json', 'w') as f_out:
    json.dump(label2id, f_out, indent=4)


    