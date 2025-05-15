import os
import vmtk
from vmtk import vmtkscripts
import glob
from tqdm import tqdm


def write_msh_single(ifile, ofile):
    try:
        # read vtu
        mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
        mesh_reader.InputFileName = ifile
        mesh_reader.Execute()
        # write mesh as fluent file
        writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
        writer.Mesh = mesh_reader.Mesh
        writer.OutputFileName = ofile
        writer.Execute()
        return True
    except Exception as e:
        print(f"Error converting {ifile}: {e}")
        return False


def process_folder(vtu_input_folder, msh_output_folder):
    """Process all VTU files in a folder and convert them to MSH format"""
    # Create output folder if it doesn't exist
    os.makedirs(msh_output_folder, exist_ok=True)
    
    # Get all VTU files in the input folder
    input_files = glob.glob(os.path.join(vtu_input_folder, "*.vtu"))
    
    if not input_files:
        print(f"No .vtu files found in {vtu_input_folder}")
        return
    
    print(f"Found {len(input_files)} VTU files to convert")
    
    # Process each file with a progress bar
    successful = 0
    for input_file in tqdm(input_files, desc="Converting VTU to MSH"):
        # Get the base filename without extension
        filename = os.path.basename(input_file)
        base_filename = os.path.splitext(filename)[0]
        
        # Create output path with MSH extension
        output_file = os.path.join(msh_output_folder, f"{base_filename}.msh")
        
        # Process the file
        if write_msh_single(input_file, output_file):
            successful += 1
    
    print(f"Successfully converted {successful}/{len(input_files)} files")


if __name__ == "__main__":
    # Define input and output folders
    vtu_input_folder = "vtu_sorted_folder"  # Folder containing input VTU files
    msh_output_folder = "msh_folder"  # Folder for output MSH files
    
    process_folder(vtu_input_folder, msh_output_folder)