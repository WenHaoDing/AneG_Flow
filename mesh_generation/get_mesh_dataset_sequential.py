from vmtk_cfdmesher import cfdmesher_single, cfdmesher_custom
import os, readline
import vmtk
from vmtk import vmtkscripts
import numpy as np
import vtk
from vtk.vtkCommonCore import vtkObject
from vtk.util import numpy_support as ns
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from vmtk_vtu2msh import write_msh_single
import pandas as pd
import pyvista as pv
import random


def sort_parts(mesh_file, visualize=False):
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
    
    # overwrite
    np2mesh = vmtk.vmtknumpytomesh.vmtkNumpyToMesh()
    np2mesh.ArrayDict = mesh_arr
    np2mesh.Execute()
    writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
    writer.Mesh = np2mesh.Mesh
    writer.OutputFileName = mesh_file
    writer.Execute()


    # record points
    for cell_id in id_set:
        cell_indices = np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]
        for cell_indice in cell_indices:
            points = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_indice).GetPoints().GetData()))
            point_set[str(cell_id)].append(points)
        point_set[str(cell_id)] = np.concatenate(point_set[str(cell_id)], axis=0)

    # visualize
    if visualize:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b']
        for i, color in zip(id_set, colors):
            points = point_set[str(i)]
            cpcd_points = cpcd_glo_gen.reshape(-1, 3)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
            ax.scatter(cpcd_points[:, 0], cpcd_points[:, 1], cpcd_points[:, 2], c='gray')
        plt.savefig(os.path.join(os.path.dirname(mesh_file), "sorted_parts.png"))


def scan_inlet_nodes(mesh_file, scale_factor=0.001):
    if scale_factor != 1:
        print(f"Scale factor is set to {scale_factor}")
    """
    Fluent parabolic inlet udf requires a csv file containing inlet node coordinates.
    This function scans the mesh and write the node coordinates into a csv file.
    """
    # check if csv exists
    csv_path = os.path.join(os.path.dirname(mesh_file), "inlet_centroids.csv")
    if os.path.exists(csv_path):
        return None
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

    # get inlet node coordinates
    cell_id = 2
    cell_indices = np.where(mesh_arr['CellData']['CellEntityIds']==cell_id)[0]
    point_set = []
    for cell_indice in cell_indices:
        points = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_indice).GetPoints().GetData()))
        point_set.append(points)
    point_set = np.concatenate(point_set, axis=0) * scale_factor
    # write to csv
    df = pd.DataFrame(point_set, columns=['x', 'y', 'z'])
    df.to_csv(csv_path, index=False)
    return None


if __name__ == "__main__":
    # conf
    # root = os.path.join(os.getcwd(), "AneuG_CFD/stable_64_v1" ) # change this to relative path on your workstation
    root = os.path.join("/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/stable_64_v3_pool" ) # change this to relative path on your workstation
    unit_factor = 1
    edge = 0.13 * unit_factor
    max_edge = 1.0 * unit_factor
    inflation = "y"
    obj_prefix = "shape_remeshed"
    vtp_prefix = "shape_remeshed"
    smoothed_vtp_prefix = vtp_prefix + "_remeshed"
    vtu_prefix = "mesh"
    msh_prefix = "mesh"
    force_scan_inlet_nodes = False  # if True, scan folders for inlet node coordinate csv files (required by udf)
    log_file = os.path.join(root, "log.txt")
    # create vtp
    create_vtp = True
    if create_vtp:
        src_files = [os.path.join(root, f, obj_prefix+".obj") for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and not os.path.exists(os.path.join(root, f, vtp_prefix+".vtp"))]
        print(src_files)
        for src in tqdm(src_files, total=len(src_files)):
            tgt = os.path.join(os.path.dirname(src), vtp_prefix+".vtp")
            pv_mesh = pv.read(src)
            pv_mesh.save(tgt)

    # load log file (with failed cases)
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            failed_paths = set(line.strip() for line in f)
    else:
        failed_paths = set()
    print("Failed cases: ", failed_paths)

    # meshing
    sequential = True
    i_start = 4096
    i_end = 170000
    if sequential:
        src_files = [os.path.join(root, f, vtp_prefix+".vtp") for f in ["stable_"+str(i) for i in range(4096, 170000)] if os.path.isdir(os.path.join(root, f))]
    else:
        src_files = [os.path.join(root, f, vtp_prefix+".vtp") for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        src_files = random.sample(src_files, len(src_files))

    readline.set_completer_delims(" \t\n=")
    readline.parse_and_bind("tab: complete")

    for src in tqdm(src_files, total=len(src_files)):
        # try remesh surface mesh
        # smoothed_vtp_path = os.path.join(os.path.dirname(src), smoothed_vtp_prefix + ".vtp")
        # arg = (f"vmtksurfacesmoothing -ifile {src} -passband 0.1 -iterations 30 -ofile {smoothed_vtp_path}")
        # os.system(arg)
        # skip if case has been tested failed
        if src in failed_paths:
            print("Skipping failed case: ", src)
            continue

        vtu_path = os.path.join(os.path.dirname(src), vtu_prefix + ".vtu")
        
        msh_path = os.path.join(os.path.dirname(src), msh_prefix + ".msh")
        # generate volume mesh
        if not os.path.exists(vtu_path):
            # cfdmesher_custom(src, vtu_path, edge, max_edge, inflation)
            cfdmesher_custom(src, vtu_path, edge, max_edge, inflation)
            
        # scan inlet nodes
        if not os.path.exists(os.path.join(os.path.dirname(vtu_path), "inlet_centroids.csv")) or force_scan_inlet_nodes:
            try:
                # sort part indices
                sort_parts(mesh_file=vtu_path)
                scan_inlet_nodes(vtu_path)
            except Exception as e:
                print("Error scanning inlet nodes: ", e)
                continue
        else:
            print("Inlet node coordinates already scanned.")
        # generate .msh file
        if not os.path.exists(msh_path):
            try: 
                write_msh_single(ifile=vtu_path, ofile=msh_path)
            except Exception as e:
                print("Error writing msh file: ", e)
                with open(log_file, "a") as f:
                    f.write(src + "\n")
                continue

"""
conda activate vmtk_add
cd VMTK
python get_mesh_dataset_custom.py

git config --global user.name "WenHaoDing"
git config --global user.email "wd123@ic.ac.uk"


ssh-keygen -t ed25519 -C "w.ding23@imperial.ac.uk"


scp -r /home/wenhao/AneuG_CFD/VMTK/AneuG/stable_64_v1 user@100.64.55.123:/F:/scp

find AneuG/stable_64 -type f -name "*.vtu" -delete
find AneuG/stable_64 -type f -name "*.msh" -delete
find AneuG/stable_64 -type f -name "*.csv" -delete
find AneuG/stable_64_v1 -type f -name "*.vtu" -delete

scp /E:/AneuG_Auto/automation_fluent/AneuG/datasets/stable_64_v2.rar meifeng@100.109.219.89:/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4

find AneuG/stable_64_v1 -type f -name "*.msh" | wc -l

scp /E:/AneuG_Auto/automation_fluent/AneuG/datasets/stable_64_v2.rar meifeng@100.109.219.89:/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4
scp -r /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/stable_64_v3_p1 user@100.64.55.123:/F:/scp

rsync /media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/stable_64_v3_p1 wenhao@100.101.90.86:/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/

rsync -av --progress -e "ssh -p 2222" \
/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/stable_64_v3_p1/ \
wenhao@100.101.90.86:/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/

rsync -r \
/media/meifeng/b1a86f8c-b396-48b9-9666-2c6b304e43d4/AneuG_CFD/stable_64_v3_p1/ \
wenhao@100.101.90.86:/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/stable_64_v3_p1
"""




