import os
import vmtk
import glob
from tqdm import tqdm

def cfdmesher_single(ifile, ofile, edge, max_edge, bl):
    # meshing with inflation layers
    if(bl=="y"):
        arg = (
            f" vmtkmeshgenerator -ifile {ifile} "
            f" -edgelength {edge} -maxedgelength {max_edge} -boundarylayer 1 -thicknessfactor 0.5 -sublayers 4 -sublayerratio 0.8 "
            f" -boundarylayeroncaps 0 -tetrahedralize 1 -ofile {ofile}"
        )
        os.system(arg)
    else:
        arg = (
            f" vmtkmeshgenerator -ifile {ifile} -edgelength {edge} -ofile {ofile}"
        )
        os.system(arg)


def process_folder(input_folder, output_folder, file_extension, edge, max_edge, bl):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files with the specified extension in the input folder
    input_files = glob.glob(os.path.join(input_folder, f"*{file_extension}"))
    
    if not input_files:
        print(f"No files with extension {file_extension} found in {input_folder}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for input_file in tqdm(input_files, desc="Processing meshes"):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder, filename.replace(file_extension, "_mesh.vtu"))
        cfdmesher_single(input_file, output_file, edge, max_edge, bl)


if __name__=="__main__":
    # Define parameters directly instead of using input()
    input_folder = "vtp_smooth_folder" 
    output_folder = "vtu_smooth_folder" 
    file_extension = ".vtp" 
    edge = "0.13"
    max_edge = "1.0"
    bl = "y"
    
    process_folder(input_folder, output_folder, file_extension, edge, max_edge, bl)