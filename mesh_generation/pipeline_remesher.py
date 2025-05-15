from vmtk_cfdmesher import cfdmesher_single
import os, readline
from tqdm import tqdm


if __name__ == "__main__":
    # conf
    root = "AneuG/datasets/stable_64"
    edge = 0.2
    inflation = "y"

    # meshing
    src_files = [os.path.join(root, f, "shape.vtp") for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    dst_files = [os.path.join(root, f, "mesh.vtu") for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    
    readline.set_completer_delims(" \t\n=")
    readline.parse_and_bind("tab: complete")

    for src, dst in tqdm(zip(src_files, dst_files), total=len(src_files)):
        cfdmesher_single(src, dst, edge, inflation)

"""
python pipeline_remesher.py
"""