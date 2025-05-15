import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from typing import Optional, Tuple
import os
from tqdm import tqdm
from typing import List, Dict
from AneuG.models.ghd_reconstruct import GHD_Reconstruct
from torch.utils.data import Dataset, DataLoader
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from AneuG.models.mesh_plugins import find_faces
from AneuG.ghd.base.graph_harmonic_deformation import mix_laplacian
from torch_geometric.nn.unpool import knn_interpolate

from .mesh_ops import generate_transform_matrices
try:
    from psbody.mesh import Mesh
except:
    print("psbody.mesh not installed, please install it to use this function.")


"""Function which reads ASCII format of Fluent solution data."""
def read_fluent_ascii(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract column names (assuming the first line contains them)
    if len(lines) <= 500:
        print("{} is empty or does not exist.".format(filename))
        tensor_data = None
        column_names = None
    else:
        column_names = lines[0].split()
        # Read numeric data, skipping the first line
        data = []
        for line in lines[1:]:
            values = list(map(float, line.split()))
            data.append(values)
        
        # Convert to a PyTorch tensor
        tensor_data = torch.tensor(np.array(data), dtype=torch.float32)
    return tensor_data, column_names


def get_fluent_ascii_raw_data(root, labels=None, write_path=None, overwrite=False, label_mapping=None,
                              threshold = 500, residual_threshold=[0.0005, 0.0005, 0.0005, 0.0005]) -> List:
    if overwrite or not os.path.exists(write_path):
        print("Reading Fluent ASCII data from", root)
        cases = [dir for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]
        data_list = []
        failed_cases = []
        for case in tqdm(cases):
            # load wall data from fluent ascii
            file_path = os.path.join(root, case, "wall")
            if not os.path.exists(file_path):
                failed_cases.append(case)
                continue
            tensors, labels_ = read_fluent_ascii(file_path)
            label_indices = None
            if labels is not None and labels_ is not None and tensors is not None:
                if all(label in labels_ for label in labels):
                    label_indices = [labels_.index(label) for label in labels]
            if label_indices is not None and len(label_indices) == len(labels):
                tensors = tensors[:, label_indices]
            else:
                failed_cases.append(case)
                continue
            # check exploded cases
            if tensors.abs().max() > threshold:
                failed_cases.append(case)
                # print("Diverged case:", case)
                continue        
            # load ghd checkpoints if exist
            ghd_chk_path = os.path.join(root, case, "checkpoint.npy")
            if os.path.exists(ghd_chk_path):
                ghd_chk = np.load(ghd_chk_path, allow_pickle=True).item()
                ghd_chk = {key: torch.tensor(value) for key, value in ghd_chk.items()}
            else:
                failed_cases.append(case)
                print("GHD checkpoint not found for case:", case)
                continue
            # load log if exist
            log_path = os.path.join(root, case, "output")
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    log_data = log_file.read()
                # residuals = np.log10(np.array(extract_last_fluent_residuals(log_data)))
                residuals = np.array(extract_last_fluent_residuals(log_data))
                if not all(residuals < residual_threshold):
                    failed_cases.append(case)
                    # print("Residuals not converged enough for case:", case)
                    continue
            else:
                log_data = None
            pair = {'label': labels if not label_mapping else label_mapping, 'tensor': tensors, 'case': case, 'ghd': ghd_chk, 'log': log_data}
            data_list.append(pair)
        torch.save({"data_list": data_list, "failed_cases": failed_cases}, write_path)
    else:
        data = torch.load(write_path)
        data_list = data['data_list']
        failed_cases = data['failed_cases']
        print("Loaded dataset from", write_path, "\n cases:", len(data_list))
    print("{} cases have been loaded successfully.".format(len(data_list)))
    return data_list, failed_cases

# def subdivide_warped_ghd_meshes(root, )


def register_raw_data_to_image(data_list, labels=None, ghd_reconstruct: GHD_Reconstruct=None, normalize=True, write_path=None, overwrite=False, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(write_path) or overwrite:
        if labels is None:
            labels = ["x", "y", "z", "wss_x", "wss_y", "wss_z"]
        # create a fluent data Meshes without connection
        data_labels = data_list[0]['label']
        fluent_data_list = [tensor[:, [data_labels.index(label) for label in labels]] for tensor in [data['tensor'] for data in data_list]]
        fluent_Meshes = Meshes(verts=[data[:, :3] for data in fluent_data_list],
                            faces=[torch.tensor([[0, 1, 2], [2, 3,
                             0]], device=fluent_data_list[0].device) for _ in fluent_data_list])
        # create a ghd data Meshes (only coordiantes)
        ghd_mean, ghd_std = data_list[0]['ghd']['mean_ghd'], data_list[0]['ghd']['std_ghd']
        ghd = torch.stack([data['ghd']['ghd'] for data in data_list], dim=0)
        norm_canonical = data_list[0]['ghd']['norm_canonical']
        scale = torch.ones(ghd.shape[0], 1, device=ghd.device) * 0.001 * norm_canonical
        ghd_Meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=ghd_mean, std=ghd_std, scale=scale)
        # extract image pixel idx using knn
        with torch.no_grad():
            # swap padded points with 1000 to avoid finding padded points
            fluent_Meshes_coords = fluent_Meshes.verts_padded().to(device)
            mask = (fluent_Meshes_coords == torch.tensor([0.0, 0.0, 0.0])).all(dim=-1).to(device)
            fluent_Meshes_coords[mask] = torch.tensor([1000.0, 1000.0, 1000.0]).to(device)
            _, idx, _ = knn_points(ghd_Meshes.verts_padded().to(device), fluent_Meshes_coords, K=1, return_nn=True)
        idx = idx.cpu()
        # form images
        B, N, _ = idx.shape
        images = torch.zeros(B, N, len(labels), device=fluent_data_list[0].device)
        ghd_dict = {'ghd': ghd, 'mean_ghd': ghd_mean, 'std_ghd': ghd_std}
        for b in range(B):
            images[b] = fluent_data_list[b][idx[b, :, 0]]
        if normalize:
            image_mean = images.mean(dim=(0, 1), keepdim=True)
            image_std = images.std(dim=(0, 1), keepdim=True)
            images = (images - image_mean) / (image_std + 1e-5)
            image_norm = {'mean': image_mean[..., 3:], 'std': image_std[..., 3:], 'mean_pcd': image_mean[..., :3], 'std_pcd': image_std[..., :3]}
        else:
            image_norm = None
        torch.save({'images': images, 'ghd_dict': ghd_dict, 'image_norm': image_norm}, write_path)
    else:
        data = torch.load(write_path)
        images = data['images']
        ghd_dict = data['ghd_dict']
        image_norm = data['image_norm']
        print("Loaded dataset from", write_path, "\n cases:", images.shape[0])
    return images, ghd_dict, image_norm


# def register_raw_data_to_image_4k(data_list, labels=None, ghd_reconstruct: GHD_Reconstruct=None, normalize=True, write_path=None, overwrite=False,
#                                   n_subdivide=2, mask=None, insert_normals=True, downsampling_factors=None):
#     if not os.path.exists(write_path) or overwrite:
#         if labels is None:
#             labels = ["x", "y", "z", "wss_x", "wss_y", "wss_z"]
#         # create a fluent data Meshes without connection
#         data_labels = data_list[0]['label']
#         x = torch.cat([tensor[:, [data_labels.index(label) for label in labels]] for tensor in [data['tensor'] for data in data_list]])
#         x = x[:, 3:]
#         pos_x = torch.cat([tensor[:, :3] for tensor in [data['tensor'] for data in data_list]], dim=0)
#         batch_x = torch.cat([torch.full((data['tensor'].shape[0],), i, dtype=torch.long) for i, data in enumerate(data_list)])
#         # fluent_data_list = [tensor[:, [data_labels.index(label) for label in labels]] for tensor in [data['tensor'] for data in data_list]]
#         # fluent_Meshes = Meshes(verts=[data[:, :3] for data in fluent_data_list],
#         #                     faces=[torch.tensor([[0, 1, 2], [2, 3,
#         #                      0]], device=fluent_data_list[0].device) for _ in fluent_data_list])

#         # create a ghd data Meshes (only coordiantes)
#         ghd_mean, ghd_std = data_list[0]['ghd']['mean_ghd'], data_list[0]['ghd']['std_ghd']
#         ghd = torch.stack([data['ghd']['ghd'] for data in data_list], dim=0)
#         norm_canonical = data_list[0]['ghd']['norm_canonical']
#         scale = torch.ones(ghd.shape[0], 1, device=ghd.device) * 0.001 * norm_canonical
#         ghd_Meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=ghd_mean, std=ghd_std, scale=scale)
#         # ghd_Meshes = clean_ghd_Meshes(ghd_Meshes, mask)
#         ghd_Meshes = clean_meshes(ghd_Meshes, mask)
#         # mask ghd_Meshes (get rid of opening vertices)
#         B = ghd_Meshes.verts_padded().shape[0]
#         # subdivide ghd_Meshes
#         with torch.no_grad():
#             ghd_Meshes = ghd_Meshes.cuda()
#             Subdivide = SubdivideMeshes(ghd_Meshes)
#             for i in range(n_subdivide):
#                 ghd_Meshes = Subdivide.forward(ghd_Meshes)
#             print("Subdivided Mesh resolution:", ghd_Meshes.verts_padded().shape[1])
#             # record subdivided mesh connectivity
#             subdivided_canonical = ghd_Meshes[0]
#             idx_list, edge_index_list = get_downsampling_edge_index(subdivided_canonical, factors=downsampling_factors)
#             pos_y = ghd_Meshes.verts_padded()
#             normals_y = ghd_Meshes.verts_normals_padded()
#             N = pos_y.shape[1]
#             pos_y = pos_y.view(-1, 3)
#             batch_y = torch.cat([torch.full((N,), i, dtype=torch.long) for i in range(B)])
#             # knn interpolate
#             images = knn_interpolate(x.cpu(), pos_x.cpu(), pos_y.cpu(), batch_x=batch_x.cpu(), batch_y=batch_y.cpu(), k=3, num_workers=1)
#             images = images.view(B, N, -1)
#             if insert_normals:
#                 images = torch.cat([pos_y.to(images.device).view(B, N, 3), normals_y.to(images.device), images], dim=-1)
#                 labels = labels[:3] + ["x_normal", "y_normal", "z_normal"] + labels[3:]
#             else:
#                 images = torch.cat([pos_y.to(images.device).view(B, N, 3), images], dim=-1)
#         if normalize:
#             # shape centering
#             images[..., :3] -= images[..., :3].mean(dim=(0, 1), keepdim=True)
#             image_mean = images.mean(dim=(0, 1), keepdim=True)
#             image_std = images.std(dim=(0, 1), keepdim=True)
#             # force scaling of shape
#             radius = torch.norm(images[..., :3], dim=-1, keepdim=True)
#             radius_mean = radius.mean()
#             radius_std = radius.std()
#             # image_mean[..., :3] = images[..., :3].mean(dim=(0, 1), keepdim=True)
#             image_std[..., :3] = radius_std + radius_mean
#             if insert_normals:
#                 image_mean[..., 3:3+3] = 0
#                 image_std[..., 3:3+3] = 1
#             images = (images - image_mean) / (image_std + 1e-5)
#             image_norm = {'mean': image_mean, 'std': image_std}
#         else:
#             image_norm = None
#         ghd_dict = {'ghd': ghd, 'mean_ghd': ghd_mean, 'std_ghd': ghd_std}
#         data = {'images': images, 'ghd_dict': ghd_dict, 'image_norm': image_norm, 'labels': labels, 'subdivided_canonical': subdivided_canonical,
#                     'idx_list': idx_list, 'edge_index_list': edge_index_list, 'downsampling_factors': downsampling_factors}
#         torch.save(data, write_path)
#     else:
#         data = torch.load(write_path)
#         print("Loaded dataset, Keys:", data.keys())
#     return data



def get_inverse_ghd_modes(GHD_eigvec):
    inverse_ghd_modes = torch.linalg.pinv(GHD_eigvec)  # [num_modes, num_pts]
    return inverse_ghd_modes



# def clean_ghd_Meshes(ghd_Meshes, mask):
#     """
#     Clean the GHD Meshes by removing vertices that are not in the mask.
#     ghd_Meshes: Meshes object
#     mask: boolean mask of shape [num_verts]
#     """
#     # Get the vertices and faces of the GHD Meshes
#     verts = ghd_Meshes.verts_padded()
#     faces = ghd_Meshes.faces_padded()
#     N = verts.shape[1]
#     B = verts.shape[0]
#     reverse_mask = torch.ones(N, dtype=torch.bool, device=mask.device)
#     reverse_mask[mask] = False
#     reverse_mask = torch.arange(N, device=mask.device)[reverse_mask]
#     # Apply the mask to the vertices
#     verts_cleaned = verts[:, mask, :]
#     # Find the faces that are still valid after applying the mask
#     faces_cleaned = find_faces(faces[0], reverse_mask)
#     faces_cleaned = faces[0][faces_cleaned]
#     # Create a new Meshes object with the cleaned vertices and faces
#     ghd_Meshes_cleaned = Meshes(verts=verts_cleaned, faces=faces_cleaned.unsqueeze(0).repeat(B, 1, 1))
#     return ghd_Meshes_cleaned


def clean_meshes(meshes: Meshes, keep_indices: torch.Tensor) -> Meshes:
    """
    Clean a batched Meshes object by keeping only vertices in `keep_indices` and
    updating faces accordingly.

    Args:
        meshes: A PyTorch3D Meshes object with padded format.
        keep_indices: A 1D LongTensor of indices to keep (shared across batch).

    Returns:
        A new Meshes object with updated verts and faces.
    """
    assert meshes._N == len(meshes.verts_padded()), "Meshes must be batched"
    B = meshes._N
    device = meshes.device
    keep_indices = keep_indices.to(device)
    
    # Create a mapping from old vertex indices to new indices
    old_to_new = -torch.ones((meshes.num_verts_per_mesh()[0],), dtype=torch.long, device=device)
    old_to_new[keep_indices] = torch.arange(len(keep_indices), device=device)

    new_verts = []
    new_faces = []

    for b in range(B):
        verts = meshes.verts_padded()[b]
        faces = meshes.faces_padded()[b]

        # Filter vertices
        verts = verts[keep_indices]
        new_verts.append(verts)

        # Mask faces: keep only faces where all 3 vertices are in keep_indices
        face_mask = (old_to_new[faces].min(dim=-1).values >= 0)
        faces = faces[face_mask]

        # Remap face indices
        remapped_faces = old_to_new[faces]
        new_faces.append(remapped_faces)

    # Pad to batch again
    new_mesh = Meshes(verts=new_verts, faces=new_faces)
    return new_mesh


def get_mode_projections(images, modes, mask=None, normalize=True):
    """
    images: [B, N, C]
    modes: [M, N]
    """
    B, N, C = images.shape
    M, N2 = modes.shape
    assert N == N2, "images and modes must have the same number of points, but got {} and {}".format(N, N2)
    # mask images & modes
    if mask is None:
        mask =  torch.arange(N).to(images.device)
    images, modes = images[:, mask, 3:], modes[:, mask]
    # compute coefficients
    coefficients = torch.einsum("mn,bnc->bmc", modes, images)
    reconstruction = torch.einsum("mn,bmc->bnc", modes, coefficients)
    residual = images - reconstruction
    # compute statistics error
    squared_error = torch.pow(residual, 2)
    sum_squared_error = torch.sum(squared_error, dim=(1, 2))  # Shape [B,]
    sum_squared_targets = torch.sum(torch.pow(images, 2), dim=(1, 2))  # Shape [B,]
    epsilon = 1e-8
    rL2 = 100 * sum_squared_error / (sum_squared_targets + epsilon)  # Shape [B,]
    # normalize
    mean, std = coefficients.mean(dim=(0, 1), keepdim=True), coefficients.std(dim=(0, 1), keepdim=True)
    coefficients_norm = {'mean': mean, 'std': std}
    print("coefficients mean:", mean)
    print("coefficients std:", std)
    if normalize:
        coefficients = (coefficients - mean) / (std+1e-5)
    return coefficients, reconstruction, residual, coefficients_norm, rL2


def knn_interpolate_p3d(x, pos_x, pos_y, k=3):
    """
    x: (B, N, C) features to interpolate from
    pos_x: (B, N, 3) coordinates of x features
    pos_y: (B, M, 3) coordinates to interpolate to
    k: number of nearest neighbors
    Returns: (B, M, C) interpolated features
    """
    _, knn_idx, _ = knn_points(pos_y, pos_x, K=k)  # (B, M, K)
    
    # Gather neighbors and compute inverse distance weights
    knn_features = knn_gather(x, knn_idx)  # (B, M, K, C)
    _, _, dists = knn_points(pos_y, pos_x, K=k)  # (B, M, K)
    weights = 1.0 / (dists + 1e-8)  # inverse distance weighting
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize
    
    # Weighted sum of neighbors
    interpolated = (weights.unsqueeze(-1) * knn_features).sum(dim=2)
    return interpolated


def get_trimmed_indice(trimmed_Meshes, canonical_Meshes):
    dists = knn_points(canonical_Meshes.verts_padded(), trimmed_Meshes.verts_padded(), K=1, return_nn=False)[0]
    verts2trim = torch.where(dists.view(-1) > 1e-5)[0]
    print("Trimmed vertices:", verts2trim.shape[0])
    total_verts = canonical_Meshes.verts_padded().shape[1]
    mask = torch.ones(total_verts, dtype=torch.bool, device=canonical_Meshes.device)
    mask[verts2trim] = False
    indices = torch.arange(total_verts, device=canonical_Meshes.device)[mask]
    return indices


def extract_last_fluent_residuals(log_text: str) -> Optional[Tuple[float, float, float, float]]:
    pattern = re.compile(r"""
        ^\s*              # leading whitespace
        (\d+)\s+          # iteration number
        ([\d.eE+-]+)\s+   # continuity
        ([\d.eE+-]+)\s+   # x-velocity
        ([\d.eE+-]+)\s+   # y-velocity
        ([\d.eE+-]+)      # z-velocity
        (?:\s+|$)         # possible trailing time info or newline
    """, re.MULTILINE | re.VERBOSE)

    matches = list(pattern.finditer(log_text))
    if not matches:
        return None
    
    last = matches[-1]
    return (
        float(last.group(2)),  # continuity
        float(last.group(3)),  # x-velocity
        float(last.group(4)),  # y-velocity
        float(last.group(5)),  # z-velocity
    )


def get_downsampling_edge_index(mesh: Meshes, factors=[4, 4]):
    mesh = Mesh(
    v=mesh.verts_packed().cpu().detach().numpy(),
    f=mesh.faces_packed().cpu().detach().numpy())
    M, _, _, _ = generate_transform_matrices(mesh, factors=factors)
    idx_list = get_mesh_downsampling_idx_batch(M)
    edge_index_list, faces_list = get_edge_index_list(M)
    return idx_list, edge_index_list, faces_list


def get_mesh_downsampling_idx_batch(M, atol=1e-6):
    idx_list = []
    # Loop through meshes, starting from the second mesh
    for i in range(1, len(M)):
        v_src = torch.tensor(M[i-1].v, dtype=torch.float32)  # vertices of the previous mesh
        v_tgt = torch.tensor(M[i].v, dtype=torch.float32)   # vertices of the current mesh
        # Compute pairwise distances
        diff = v_src.unsqueeze(0) - v_tgt.unsqueeze(1)  # (N_tgt, N_src, 3)
        dists = torch.norm(diff, dim=2)                 # (N_tgt, N_src)
        # Find nearest source vertex for each target vertex
        min_dists, idx = dists.min(dim=1)               # (N_tgt,)
        # Check if any distance exceeds the tolerance
        if (min_dists > atol).any():
            raise ValueError("Some vertices in mesh_tgt do not match any vertex in mesh_src within tolerance.")
        idx_list.append(idx)
    return idx_list


def get_edge_index_list(M):
    edge_index_list = []
    faces_list = []
    for mesh in M:
        faces = torch.tensor(mesh.f.astype(np.int64), dtype=torch.long)  # (F, 3)
        edges = torch.cat([
            faces[:, [0, 1]],  # edge (i, j)
            faces[:, [1, 2]],  # edge (j, k)
            faces[:, [2, 0]]   # edge (k, i)
        ], dim=0)  # (3 * F, 2)
        edges = edges.sort(dim=1)[0]
        edges = torch.unique(edges, dim=0)  # (E, 2)
        # Add edge_index to the list
        edge_index_list.append(edges.t())  # (2, E)
        faces_list.append(faces)
    return edge_index_list, faces_list


def register_raw_data_to_image_4k(data_list, labels=None, ghd_reconstruct: GHD_Reconstruct=None, normalize=True, write_path=None, overwrite=False,
                                  n_subdivide=2, mask=None, insert_normals=True, downsampling_factors=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(write_path) or overwrite:
        if labels is None:
            labels = ["x", "y", "z", "wss_x", "wss_y", "wss_z"]
        # create a fluent data Meshes without connection
        data_labels = data_list[0]['label']
        x = torch.cat([tensor[:, [data_labels.index(label) for label in labels]] for tensor in [data['tensor'] for data in data_list]])
        x = x[:, 3:]
        pos_x = torch.cat([tensor[:, :3] for tensor in [data['tensor'] for data in data_list]], dim=0)
        batch_x = torch.cat([torch.full((data['tensor'].shape[0],), i, dtype=torch.long) for i, data in enumerate(data_list)])
        # fluent_data_list = [tensor[:, [data_labels.index(label) for label in labels]] for tensor in [data['tensor'] for data in data_list]]
        # fluent_Meshes = Meshes(verts=[data[:, :3] for data in fluent_data_list],
        #                     faces=[torch.tensor([[0, 1, 2], [2, 3,
        #                      0]], device=fluent_data_list[0].device) for _ in fluent_data_list])
        case_names = [data['case'] for data in data_list]
        # create a ghd data Meshes (only coordiantes)
        ghd_mean, ghd_std = data_list[0]['ghd']['mean_ghd'], data_list[0]['ghd']['std_ghd']
        ghd = torch.stack([data['ghd']['ghd'] for data in data_list], dim=0)
        norm_canonical = data_list[0]['ghd']['norm_canonical']
        scale = torch.ones(ghd.shape[0], 1, device=ghd.device) * 0.001 * norm_canonical
        ghd_Meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=ghd_mean, std=ghd_std, scale=scale)
        # ghd_Meshes = clean_ghd_Meshes(ghd_Meshes, mask)
        ghd_Meshes = clean_meshes(ghd_Meshes, mask)
        # mask ghd_Meshes (get rid of opening vertices)
        B = ghd_Meshes.verts_padded().shape[0]
        print("canonical Mesh resolution:", ghd_Meshes.verts_padded().shape[1])
        print("canonical Mesh faces:", ghd_Meshes.faces_padded().shape[1])
        # subdivide ghd_Meshes
        with torch.no_grad():
            ghd_Meshes = ghd_Meshes.to(device)
            Subdivide = SubdivideMeshes(ghd_Meshes)
            for i in range(n_subdivide):
                ghd_Meshes = Subdivide.forward(ghd_Meshes)
            print("Subdivided Mesh resolution:", ghd_Meshes.verts_padded().shape[1])
            print("Subdivided Mesh faces:", ghd_Meshes.faces_padded().shape[1])
            # record subdivided mesh connectivity
            subdivided_canonical = ghd_Meshes[0]
            idx_list, edge_index_list, faces_list = get_downsampling_edge_index(subdivided_canonical, factors=downsampling_factors)
            pos_y = ghd_Meshes.verts_padded()
            normals_y = ghd_Meshes.verts_normals_padded()
            N = pos_y.shape[1]
            pos_y = pos_y.view(-1, 3)
            batch_y = torch.cat([torch.full((N,), i, dtype=torch.long) for i in range(B)])
            # knn interpolate
            images = knn_interpolate(x.cpu(), pos_x.cpu(), pos_y.cpu(), batch_x=batch_x.cpu(), batch_y=batch_y.cpu(), k=3, num_workers=1)
            images = images.view(B, N, -1)
            if insert_normals:
                images = torch.cat([pos_y.to(images.device).view(B, N, 3), normals_y.to(images.device), images], dim=-1)
                labels = labels[:3] + ["x_normal", "y_normal", "z_normal"] + labels[3:]
            else:
                images = torch.cat([pos_y.to(images.device).view(B, N, 3), images], dim=-1)
        if normalize:
            # shape centering
            # images[..., :3] -= images[..., :3].mean(dim=(0, 1), keepdim=True)
            image_mean = images.mean(dim=(0, 1), keepdim=True)
            image_std = images.std(dim=(0, 1), keepdim=True)
            # force scaling of shape
            radius = torch.norm(images[..., :3], dim=-1, keepdim=True)
            radius_mean = radius.mean()
            radius_std = radius.std()
            image_mean[..., :3] = radius_mean
            image_std[..., :3] = radius_std
            if insert_normals:
                image_mean[..., 3:3+3] = 0
                image_std[..., 3:3+3] = 1
            images = (images - image_mean) / (image_std + 1e-5)
            image_norm = {'mean': image_mean, 'std': image_std}
        else:
            image_norm = None
        ghd_dict = {'ghd': ghd, 'mean_ghd': ghd_mean, 'std_ghd': ghd_std}
        data = {'images': images, 'ghd_dict': ghd_dict, 'image_norm': image_norm, 'labels': labels, 'subdivided_canonical': subdivided_canonical,
                'idx_list': idx_list, 'edge_index_list': edge_index_list, 'faces_list': faces_list, 'downsampling_factors': downsampling_factors, 'case_names': case_names}
        torch.save(data, write_path)
    else:
        data = torch.load(write_path)
        print("Loaded dataset, Keys:", data.keys())
    return data


