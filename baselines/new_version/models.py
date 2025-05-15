import os.path as osp
import torch
import torch.nn.functional as F
from typing import List
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch
from pytorch3d.structures import Meshes
import torch.nn.functional as F


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class SimplePointNet2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([6 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 6, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data):
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)

class SimplePointNet2_V2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.25, 0.2, MLP([6 + 3, 32, 32, 64]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 64, 64, 128]))
        self.sa3_module = SAModule(0.25, 0.8, MLP([128 + 3, 128, 128, 256]))
        self.sa4_module = GlobalSAModule(MLP([256 + 3, 512, 1024, 2048]))

        self.mlp = MLP([64, 32, num_classes], dropout=0.0, norm=None)

        self.fp4_module = FPModule(1, MLP([2048 + 256, 1024, 512]))
        self.fp3_module = FPModule(3, MLP([512+128, 256, 128]))
        self.fp2_module = FPModule(3, MLP([128 + 64, 128, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 6, 64, 64, 64]))

    def forward(self, data: Data):
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)


class SimplePointNet2_V3(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([6 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = SAModule(0.25, 0.8, MLP([256 + 3, 256, 512, 512]))
        self.sa4_module = GlobalSAModule(MLP([512 + 3, 512, 1024, 2048]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.0, norm=None)

        self.fp4_module = FPModule(1, MLP([2048 + 512, 1024, 512]))
        self.fp3_module = FPModule(3, MLP([512+256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 128, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 6, 128, 128, 128]))

    def forward(self, data: Data):
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)





def images_to_pyg_data(images, dataset_labels, x_labels, y_labels, device=None):
    """
    Convert images to PyG Data format.
    images: [B, N, C]
    """
    B, N, C = images.shape
    pos = images[:, :, :3]  # First 3 channels as pos
    batch = torch.arange(B, device=images.device).view(-1, 1).repeat(1, N).view(-1)
    x = images[..., :3].view(-1, C)  # Flatten images to [B * N, C]
    y = images[..., 3:].view(-1, C)  # Flatten images to [B * N, C]
    pos = pos.view(-1, 3)  # Flatten pos to [B * N, 3]
    pyg_batch = Batch(x=x, y=y, pos=pos, batch=batch)
    if device is not None:
        pyg_batch = pyg_batch.to(device)
    return pyg_batch


class ImageToPYGData(object):
    def __init__(self, dataset_labels, x_labels, y_labels, device=None):
        self.dataset_labels = dataset_labels
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.pos_labels = ['x', 'y', 'z']
        self.device = device
        self.x_idx = [dataset_labels.index(label) for label in x_labels]
        self.y_idx = [dataset_labels.index(label) for label in y_labels]
        self.pos_idx = [dataset_labels.index(label) for label in self.pos_labels]
    
    # def forward(self, images):
    #     B, N, C = images.shape
    #     pos = images[:, :,self.pos_idx] 
    #     batch = torch.arange(B, device=images.device).view(-1, 1).repeat(1, N).view(-1)
    #     x = images[..., self.x_idx].view(-1, len(self.x_idx))  # Flatten images to [B * N, C]
    #     y = images[..., self.y_idx].view(-1, len(self.y_idx))  # Flatten images to [B * N, C]
    #     pos = pos.view(-1, 3)  # Flatten pos to [B * N, 3]
    #     pyg_batch = Batch(x=x, y=y, pos=pos, batch=batch)
    #     if self.device is not None:
    #         pyg_batch = pyg_batch.to(self.device)
    #     return pyg_batch, images.shape

    def forward(self, images):
        B, N, C = images.shape
        pos = images[..., self.pos_idx] 
        x = images[..., self.x_idx]
        y = images[..., self.y_idx]
        pyg_data_list = [Data(x=x[i], y=y[i], pos=pos[i]) for i in range(B)]
        pyg_batch = Batch.from_data_list(pyg_data_list)
        if self.device is not None:
            pyg_batch = pyg_batch.to(self.device)
        return pyg_batch, images.shape




