import os
import vmtk
from vmtk import vmtkscripts
import numpy as np
import vtk
from vtkmodules.vtkCommonCore import vtkObject
from vtkmodules.util import numpy_support as ns
import copy
import glob
from tqdm import tqdm


def sort_parts(mesh_file, checkpoint_file, output_file):
    try:
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
        
        # load checkpoint
        checkpoint = np.load(checkpoint_file, allow_pickle=True).item()
        cpcd_glo_gen = checkpoint["cpcd_glo_gen"] * checkpoint['norm_canonical']

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
            if point_set[str(cell_id)]:  # Check if list is not empty
                point_set[str(cell_id)] = np.concatenate(point_set[str(cell_id)], axis=0)

        # save sorted mesh
        np2mesh = vmtk.vmtknumpytomesh.vmtkNumpyToMesh()
        np2mesh.ArrayDict = mesh_arr
        np2mesh.Execute()

        # write mesh as vtu file
        mesh_writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
        mesh_writer.Mesh = np2mesh.Mesh
        mesh_writer.OutputFileName = output_file
        mesh_writer.Execute()
        
        return True
    except Exception as e:
        print(f"Error processing {mesh_file}: {e}")
        return False


def process_folder(vtu_input_folder, checkpoint_folder, sorted_output_folder):
    """Process all VTU files using checkpoints from another folder"""
    # Create output folder if it doesn't exist
    os.makedirs(sorted_output_folder, exist_ok=True)
    
    # Get all VTU files in the input folder
    input_files = glob.glob(os.path.join(vtu_input_folder, "*.vtu"))
    
    if not input_files:
        print(f"No .vtu files found in {vtu_input_folder}")
        return
    
    print(f"Found {len(input_files)} VTU files to process")
    
    # Process each file with a progress bar
    successful = 0
    for input_file in tqdm(input_files, desc="Sorting mesh parts"):
        # Get the base filename without extension
        filename = os.path.basename(input_file)
        base_filename = os.path.splitext(filename)[0]
        case_id = base_filename.split('_')[0]
        
        # Construct checkpoint path with matching filename
        checkpoint_file = os.path.join(checkpoint_folder, f"{case_id}.npy")
        if not os.path.exists(checkpoint_file):
            print(f"Warning: No checkpoint found for {filename} (looked for {case_id}.npy)")
            continue
        
        # Output path
        output_file = os.path.join(sorted_output_folder, f"{base_filename}_sorted.vtu")
        
        # Process the file
        if sort_parts(input_file, checkpoint_file, output_file):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(input_files)} files")


if __name__ == "__main__":
    # Define input, checkpoint, and output folders
    vtu_input_folder = "vtu_folder"  # Folder containing input VTU files
    checkpoint_folder = "checkpoint_folder"  # Folder containing checkpoint files
    sorted_output_folder = "vtu_sorted_folder"  # Folder for sorted output files
    
    process_folder(vtu_input_folder, checkpoint_folder, sorted_output_folder)