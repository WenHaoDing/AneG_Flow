import os
import vmtk
import glob
from tqdm import tqdm

def smooth_surface(input_file, output_file, passband, iterations):
    command = f"vmtksurfacesmoothing -ifile {input_file} -passband {passband} -iterations {iterations} -ofile {output_file}"
    os.system(command)

def process_folder(input_folder, output_folder, passband, iterations):
    """Process all VTP files in a folder, applying smoothing to each"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all VTP files in the input folder
    input_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    
    if not input_files:
        print(f"No .vtp files found in {input_folder}")
        return
    
    print(f"Found {len(input_files)} VTP files to process")
    
    # Process each file with a progress bar
    for input_file in tqdm(input_files, desc="Smoothing surfaces"):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder, filename.replace(".vtp", "_smooth.vtp"))
        smooth_surface(input_file, output_file, passband, iterations)

if __name__ == "__main__":
    # Define parameters directly
    input_folder = "vtp_folder" 
    output_folder = "vtp_smooth_folder"  
    passband = 0.1  # Smoothing passband parameter
    iterations = 30  # Number of smoothing iterations
    
    process_folder(input_folder, output_folder, passband, iterations)