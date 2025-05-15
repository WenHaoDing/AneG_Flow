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
from torch_geometric.nn import SplineConv
import torch.nn.functional as F


class SplineCNNSAModule(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dim=3, n_layers=3, kernel_size=5, patch_size=0.25, degree=1, drop_act=False):
        super().__init__()
        self.patch_size = patch_size
        self.conv = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.drop_act = drop_act
        assert n_layers >= 2, 'Number of layers should be at least 2'
        for i in range(n_layers):
            if i == 0:
                self.conv.append(SplineConv(in_channels, hidden_channels, dim=dim, kernel_size=kernel_size, degree=degree))
            elif i == n_layers - 1:
                self.conv.append(SplineConv(hidden_channels, out_channels, dim=dim, kernel_size=kernel_size, degree=degree))
            else:
                self.conv.append(SplineConv(hidden_channels, hidden_channels, dim=dim, kernel_size=kernel_size, degree=degree))

    def forward(self, x, pos, batch, idx, edge_index):
        # convolution
        pseudo = (pos[edge_index[1]] - pos[edge_index[0]]) / self.patch_size + 0.5
        pseudo = pseudo.clamp(min=0, max=1)
        for i in range(self.n_layers):
            if (i == self.n_layers - 1) and self.drop_act:
                x = self.conv[i](x, edge_index, pseudo)                
            else:
                x = F.elu(self.conv[i](x, edge_index, pseudo))
        # pooling
        if idx is not None:
            x, pos, batch = x[idx], pos[idx], batch[idx]
        return x, pos, batch


class SplineCNNFPModule(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dim=3, n_layers=3, kernel_size=5, patch_size=0.25, degree=1, drop_act=False, k=3):
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.conv = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.drop_act = drop_act
        assert n_layers >= 2, 'Number of layers should be at least 2'
        for i in range(n_layers):
            if i == 0:
                self.conv.append(SplineConv(in_channels, hidden_channels, dim=dim, kernel_size=kernel_size, degree=degree))
            elif i == n_layers - 1:
                self.conv.append(SplineConv(hidden_channels, out_channels, dim=dim, kernel_size=kernel_size, degree=degree))
            else:
                self.conv.append(SplineConv(hidden_channels, hidden_channels, dim=dim, kernel_size=kernel_size, degree=degree))

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip, edge_index):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        pseudo = (pos_skip[edge_index[1]] - pos_skip[edge_index[0]]) / self.patch_size + 0.5
        pseudo = pseudo.clamp(min=0, max=1)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        for i in range(self.n_layers):
            if (i == self.n_layers - 1) and self.drop_act:
                x = self.conv[i](x, edge_index, pseudo)                
            else:
                x = F.elu(self.conv[i](x, edge_index, pseudo))
        return x, pos_skip, batch_skip


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        # upsample to previous resolution
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
        pos, batch = pos_skip, batch_skip
        x = torch.cat([x, x_skip], dim=1)
        return x, pos, batch


class SplineCNNUNet_V1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, idx_list, edge_index_list, patch_size_list, 
                 kernel_size=5, degree=1):
        super().__init__()
        self.num_layers = len(idx_list)
        self.idx_list = idx_list
        self.edge_index_list = edge_index_list
        self.patch_size_list = patch_size_list

        self.sa1_module = SplineCNNSAModule(in_channels, 16, 32, 3, 3, kernel_size, patch_size_list[0], degree)
        self.sa2_module = SplineCNNSAModule(32, 64, 128, 3, 3, kernel_size, patch_size_list[1], degree)

        self.bn_module = SplineCNNSAModule(128, 128, 128, 3, 3, kernel_size, patch_size_list[2], degree)

        self.gp1_module = GlobalSAModule(MLP([128 + 3, 256, 512, 1024]))

        self.fp2_module = SplineCNNFPModule(1024+128 + 32, 256, 128, 3, 3, kernel_size, patch_size_list[1], degree)
        self.fp1_module = SplineCNNFPModule(128 + 6, 128, 128, 3, 3, kernel_size, patch_size_list[0], degree)

        self.mlp = MLP([128, 64, out_channels], norm=None)
    
    
    def forward(self, data):
        idx_list, edge_index_list = duplicate_edge_index(data, self.edge_index_list, self.idx_list)

        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out, idx_list[0], edge_index_list[0])
        sa2_out = self.sa2_module(*sa1_out, idx_list[1], edge_index_list[1])
        
        bn_out = self.bn_module(*sa2_out, None, edge_index_list[2])

        sa2_out = self.gp1_module(*sa2_out, *bn_out)
        fp2_out = self.fp2_module(*sa2_out, *sa1_out, edge_index_list[1])
        fp1_out = self.fp1_module(*fp2_out, *sa0_out, edge_index_list[0])
        x, _, _ = fp1_out  # print(fp1_out[0].isnan().sum())
        out = self.mlp(x)
        return out


def duplicate_edge_index(data, edge_index_list, idx_list):
        """
        edge_index = torch.Tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).long()
        # print(edge_index)
        B = 3
        n_node = 5
        shift_edge_index = (torch.arange(B) * n_node).unsqueeze(1).repeat(1, edge_index.shape[1]).view(-1).unsqueeze(0)
        duplicated_edge_index = edge_index.permute(1, 0).unsqueeze(0).repeat(B, 1, 1).view(-1, 2).permute(1, 0)
        print(duplicated_edge_index.view(2, B, -1)[:, 0, :])
        # print(duplicated_edge_index)
        # print(shift_edge_index)
        print(duplicated_edge_index + shift_edge_index)
        """
        # mind the sequence arrangement
        B = data.batch.max().item() + 1
        duplicated_idx_list = []
        duplicated_edge_index_list = []
        # N = data.x.shape[0] / B
        # assert N == int(N), f"Batch size {B} does not divide the number of nodes {data.x.shape[0]}"

        for i in range(len(edge_index_list)):
            if i != len(edge_index_list) - 1:
                idx = idx_list[i]
                N = int(data.x.shape[0] / B) if i == 0 else len(idx_list[i-1])
                shift_idx = (N * torch.arange(B)).unsqueeze(1).repeat(1, len(idx)).view(-1).to(data.x.device)
                idx = idx.to(data.x.device).unsqueeze(0).repeat(B, 1).view(-1) + shift_idx
                duplicated_idx_list.append(idx.clone().detach())
            
            if i == 0:
                 n_node = int(data.x.shape[0] / B)
            else:
                 n_node = len(idx_list[i-1])
            
            edge_index = edge_index_list[i]
            shift_edge_index = (torch.arange(B) * n_node).unsqueeze(1).repeat(1, edge_index.shape[1]).view(-1).unsqueeze(0).to(data.x.device)
            duplicated_edge_index = edge_index.permute(1, 0).unsqueeze(0).repeat(B, 1, 1).view(-1, 2).permute(1, 0).to(data.x.device)
            duplicated_edge_index = duplicated_edge_index + shift_edge_index
            duplicated_edge_index_list.append(duplicated_edge_index.clone().detach())
        return duplicated_idx_list, duplicated_edge_index_list