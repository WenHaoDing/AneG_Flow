import os
import vmtk
from vmtk import vmtkscripts
import numpy as np
import vtk
from vtk.vtkCommonCore import vtkObject
from vtk.util import numpy_support as ns
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


def sort_parts(mesh_file):
    # read vtu
    mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
    mesh_reader.InputFileName = mesh_file
    mesh_reader.Execute()
    # convert vtu to np array
    mesh2np = vmtk.vmtkmeshtonumpy.vmtkMeshToNumpy()
    mesh2np.Mesh = mesh_reader.Mesh
    mesh2np.Execute()
    mesh_arr = mesh2np.ArrayDict
    # read vtu with vtk
    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(mesh_file)
    vtk_reader.Update()
    ugrid = vtk_reader.GetOutput()
    cell_ids = np.unique(mesh_arr['CellData']['CellEntityIds'])
    print(cell_ids)
    cell_number = [len(np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]) for cell_id in cell_ids]
    print(cell_number)
    
    # load opening checkpoint
    checkpoint_path = os.path.join(os.path.dirname(mesh_file), "checkpoint.npy")
    checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
    cpcd_glo_gen = checkpoint["cpcd_glo_gen"] * checkpoint['norm_canonical']
    # cpcd_glo_gen = cpcd_glo_gen[:, -1:, :]  # [num_branch, 1, 3]

    id_set = [2, 3, 4]
    # get sort sequence
    cell_coordinates = []
    for cell_type in id_set:
        cell_id = np.where(mesh_arr['CellData']['CellEntityIds']==cell_type)[0][0]
        cell_coord = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_id).GetPoints().GetData()))
        cell_coordinates.append(cell_coord)
    cell_coordinates = np.stack(cell_coordinates, axis=0).mean(axis=1)  # [num_branch, 3]
    cell_coordinates = np.expand_dims(cell_coordinates, axis=0)  # [1, num_branch, 3]
    distances = np.linalg.norm(cell_coordinates - cpcd_glo_gen[:, -1:, :], axis=-1)  # [num_branch, num_branch]
    sort_sequence = np.argmin(distances, axis=0)

    # sort parts
    point_set = {}
    for i in id_set:
         point_set[str(i)] = []
    for i, cell_id in enumerate(mesh_arr['CellData']['CellEntityIds']):
            if cell_id in id_set:
                    original_id = id_set.index(cell_id)
                    new_id = id_set[sort_sequence[original_id]]
                    mesh_arr['CellData']['CellEntityIds'][i] = new_id
    # record points
    for cell_id in id_set:
        cell_indices = np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]
        for cell_indice in cell_indices:
            points = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_indice).GetPoints().GetData()))
            point_set[str(cell_id)].append(points)
        point_set[str(cell_id)] = np.concatenate(point_set[str(cell_id)], axis=0)

    # visualize
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i, color in zip(id_set, colors):
        points = point_set[str(i)]
        cpcd_points = cpcd_glo_gen.reshape(-1, 3)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
        ax.scatter(cpcd_points[:, 0], cpcd_points[:, 1], cpcd_points[:, 2], c='gray')
    plt.savefig(os.path.join(os.path.dirname(mesh_file), "sorted_parts.png"))


if __name__ == "__main__":
    # conf
    root = "AneuG/datasets/stable_64"
    mesh_files = [os.path.join(root, dir, "mesh.vtu") for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir)) and 
                os.path.exists(os.path.join(root, dir, "mesh.vtu"))]
    for mesh_file in tqdm(mesh_files):
        sort_parts(mesh_file)

    
"""
python pipeline_sort_parts.py

git config --global user.name "WenHaoDing"
git config --global user.email "w.ding23@imperial.ac.uk"

git config --global user.name "anonymousaneug"
git config --global user.email "anonymousaneug@gmail.com"

"""