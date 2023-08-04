import numpy as np
import os
import sys
import h5py
from open3d.io import read_point_cloud
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, Tuple, List, Literal

"""
Load cell meshes of different tissues from their directories, convert them to point clouds, and 
associate a label (usually the shape) to each one.
Finally, split data in training, validation and test sets and store them in HDF5 files for use
in the FoldingNet model.
"""


#-------------------------------------------------------------------------------------------------
def _convert_mesh_to_coordinates(
        file_path: str,
        sample_n_points: Optional[int] = 2048
) -> np.ndarray[float]:
    """
    Convert a mesh (.ply, .stl, .vtk, ...) into an array of coordinates.

    Parameters:
    -----------
    file_path: (str) 
        The path to the file to convert.

    sample_n_points: (Optional[int], default=2048)
        The number of points to sample from the mesh coordinates.

    Returns:
    --------
    pc: (np.ndarray[float])
        A np.ndarray containing the point cloud coordinates.
    """
    # get file name and file extensione
    fname = os.path.basename(file_path)
    f_ext = fname[-3:]

    assert f_ext not in ["ply", "stl", "vtk"], "The file extension should be among those: "".ply"", "".stl"", "".vtk"""

    assert sample_n_points > 0, "Cannot sample a negative number of points."

    pc = read_point_cloud(file_path, format=f_ext)

    pc = np.asarray(pc.points)

    pc = pc[np.random.choice(len(pc), sample_n_points, replace=False), :]

    return pc
#-------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
def create_point_clouds_stack(
        mesh_dir_path:  str,
        mesh_label: str,
        n_points: Optional[int] = 2048,
) -> Tuple[np.ndarray[float], np.ndarray[str]]:
    """
    Load all the mesh files from the specified directory, convert them in point cloud data and 
    stack them in numpy arrays that will be later on saved in hdf5 datasets.

    Parameters:
    -----------
    mesh_dir_path:  (str)
        The path to the directory to load the meshes from.
    
    mesh_label: (str) 
        A string defining the class of the current meshes.

    n_points: (Optional[int], default=2048)
        The number of points to sample from the mesh coordinates.
    
    Returns:
    --------
    point_clouds_stack: (np.ndarray[float])
        A numpy array of point clouds data of size (N, npoints, 3), where N is the number
        of instances, and npoints is the number of coordinates for each point cloud.

    labels_stack: (np.ndarray[str])
        A numpy array of labels data of size (N, 1), where N is the number of instances.
    """

    assert os.path.exists(mesh_dir_path), f"""\
        The specified path to the directory storing mesh files does not exist {mesh_dir_path}
        """
    
    mesh_files = os.listdir(mesh_dir_path)

    point_clouds_stack = np.empty((len(mesh_files), 2048, 3), dtype=np.float32)
    labels_stack = np.empty((len(mesh_files), 1), dtype="|S10")
    for i, mesh_file in tqdm(enumerate(mesh_files), desc="Converting meshes", total=len(mesh_files)):
        pc = _convert_mesh_to_coordinates(
            file_path=os.path.join(mesh_dir_path, mesh_file),
            sample_n_points=n_points
        )
        point_clouds_stack[i, :, :] = pc
        labels_stack[i, :] = mesh_label

    return point_clouds_stack, labels_stack
#-------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
def split_train_val_test(
        data: np.ndarray,
        ratios: Optional(List[float]) = [0.8, 0,1, 0.1],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the input array into training, validation and test sets along its first dimension.

    Parameters:
    -----------
    data: (np.ndarray)
        The input array to be splitted along its first dimension.

    ratios: (Optional[List[float]], default=[0.8, 0,1, 0.1])
        The list of training, validation and test size ratios.

    Returns:
    --------
    train_data, val_data, test_data: (np.ndarray)
        The splitted arrays.
    """

    assert sum(ratios) == 1, "Splitting ratios do no sum up to 1."

    train_ratio, val_ratio, test_ratio = ratios[0], ratios[1], ratios[2]

    train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), shuffle=True)

    return train_data, val_data, test_data
#-------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
def _split_in_substacks(
    input_stack: np.ndarray,
    substack_size: Optional[int] = 2048,
) -> List[np.ndarray[float]]:
    """
    Parameters:
    -----------
    input_stack: (np.ndarray)
        The stack to split into substacks.

    substack_size: (Optional[int], default=2048) 
        The number of point clouds to store in a single stack. 

    Returns:
    --------
    substacks_lst: (List[np.ndarray[float]])
        A collection of stacks of point clouds data, each one containing stack_size samples.
    """

    num_records = input_stack.shape[0]
    substack_lst = []
    start, end = 0, min(substack_size, num_records) 
    while end <= num_records:
        substack_lst.append(input_stack[start:end, ...])
        start += substack_size
        end = min(start + substack_size, num_records)

    return substack_lst
#-------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
def save_data(
    point_clouds_stack: List[np.ndarray[float]],
    labels_stack: List[np.ndarray[str]],
    split: Literal['train', 'val', 'test'],
    save_dir: str,
    max_records_per_file: Optional[int] = 2048,
) -> None:
    """
    Split the input stack in subsets of the desired size and save them in separate HDF5 datasets.

    Parameters:
    -----------
    point_clouds_stack: (np.ndarray[float])
        An array storing all the point clouds data for a certain split.
    
    labels_stack: (np.ndarray[str])
        An array storing all the point clouds data for a certain split.
    
    split: (Literal['train', 'val', 'test'])
        Whether the input data are part of training, validation or test set.

    save_dir: (str)
        The path to the directory to store the converted file in.

    max_records_per_file:( Optional[int] = 2048)
        The max number of records that is possible to store in a file.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    point_clouds_substack_lst = _split_in_substacks(
        input_stack=point_clouds_stack,
        substack_size=max_records_per_file
    )
    labels_substack_lst = _split_in_substacks(
        input_stack=labels_stack,
        substack_size=max_records_per_file
    )

    file_prefix = "data_" + split + "_n"
    for i in range(len(point_clouds_substack_lst)):
        file_name = file_prefix + str(i) + ".h5"
        with h5py.File(os.path.join(save_dir, file_name), "w") as f_out:
            f_out.create_dataset(
                name="point_clouds",
                data=point_clouds_substack_lst[i]
            )
            f_out.create_dataset(
                name="labels",
                data=labels_substack_lst[i]
            )

#-------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------
def main(
    mesh_dirs: List[str],
    labels: List[Literal["cuboidal", "columnar", "squamous", "pseudostratified", "transitional"]],
    split_ratios: List[float],
    npoints: int,
    path_to_save_dir: str,
    max_records_per_file: Optional[int] = 2048
) -> None:
    
    # Add data from different samples to global lists
    all_pcs_lst, all_lbls_lst = []
    for mesh_dir, label in zip(mesh_dirs, labels):
        # Convert meshed into point clouds, store them in stacks
        curr_point_clouds_stack, curr_labels_stack = create_point_clouds_stack(
            mesh_dir_path=mesh_dir, 
            mesh_label=label,
            n_points=npoints
        )        
        # Append the results to the global stacks 
        all_pcs_lst.append(curr_point_clouds_stack)
        all_lbls_lst.append(curr_labels_stack)

    # Concatenate in single arrays
    all_point_clouds_stack = np.concatenate(all_pcs_lst)
    all_labels_stack = np.concatenate(all_lbls_lst)

    # Split in train, val, test
    idxs = np.arange(all_point_clouds_stack.shape[0])
    train_idxs, val_idxs, test_idxs = split_train_val_test(data=idxs, ratios=split_ratios)
    all_point_clouds_stack_train = all_point_clouds_stack[train_idxs, ...]
    all_point_clouds_stack_val = all_point_clouds_stack[val_idxs, ...]
    all_point_clouds_stack_test = all_point_clouds_stack[test_idxs, ...]
    all_labels_stack_train = all_labels_stack[train_idxs, ...]
    all_labels_stack_val = all_labels_stack[val_idxs, ...]
    all_labels_stack_test = all_labels_stack[test_idxs, ...]

    # Save the data
    save_data(
        point_clouds_stack=all_point_clouds_stack_train,
        labels_stack=all_labels_stack_train,
        split="train",
        save_dir=path_to_save_dir,
        max_records_per_file=max_records_per_file
    )
    save_data(
        point_clouds_stack=all_point_clouds_stack_val,
        labels_stack=all_labels_stack_val,
        split="val",
        save_dir=path_to_save_dir,
        max_records_per_file=max_records_per_file
    )
    save_data(
        point_clouds_stack=all_point_clouds_stack_test,
        labels_stack=all_labels_stack_test,
        split="test",
        save_dir=path_to_save_dir,
        max_records_per_file=max_records_per_file
    )

#-------------------------------------------------------------------------------------------------



if __name__=="__main__":

    path_to_mesh_dir = "../data/CellsData/cell_meshes_bladder/"
    file_names = os.listdir(path_to_mesh_dir)
    file_paths = [os.path.join(path_to_mesh_dir, file_name) for file_name in file_names]

    for fpath in tqdm(file_paths, desc="Converting files: "):
        convert_mesh_to_coords(
            file_path=fpath, 
            label="cuboidal",
            save_dir="../data/CellsData/point_clouds_bladder",
        )

    print("All the files have been successfully converted!")
#-------------------------------------------------------------------------------------------------