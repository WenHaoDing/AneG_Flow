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
from torch_geometric.nn import SplineConv, GCNConv, NNConv, GATConv, ChebConv
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention
import math
from new_version.SplineCNNUNet import duplicate_edge_index


class SAModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, conv_type='gcn', drop_act=False, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.drop_act = drop_act
        self.conv = nn.ModuleList()
        for i in range(n_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == n_layers - 1 else hidden_channels
            self.conv.append(get_graph_conv(conv_type, in_c, out_c, **kwargs))

    def forward(self, x, pos, batch, idx, edge_index, skip_pos_encoding=False):
        x = torch.cat([x, pos], dim=1) if not skip_pos_encoding else x
        for i, conv in enumerate(self.conv):
            x = conv(x, edge_index)
            if i != self.n_layers - 1 or not self.drop_act:
                x = F.relu(x)
        # pooling
        if idx is not None:
            x, pos, batch = x[idx], pos[idx], batch[idx]
        return x, pos, batch
    

class FPModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, k=3, conv_type='gcn', drop_act=False, **kwargs):
        super().__init__()
        self.k = k
        self.drop_act = drop_act
        self.conv = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(n_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == n_layers - 1 else hidden_channels
            self.conv.append(get_graph_conv(conv_type, in_c, out_c, **kwargs))

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip, edge_index):
        # 1. Upsample features from coarse to fine resolution
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        # 2. Concatenate with skip connection (if provided)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        # 3. Apply GCN layers
        for i, conv in enumerate(self.conv):
            x = conv(x, edge_index)
            if i != self.n_layers - 1 or not self.drop_act:
                x = F.relu(x)
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
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=1)
        pos, batch = pos_skip, batch_skip
        x = torch.cat([x, x_skip], dim=1)
        return x, pos, batch

class NavieUNet_V1(nn.Module):
    def __init__(self, in_channels, out_channels, idx_list, edge_index_list, global_token_dim=2048, conv_type='gcn'):
        super().__init__()
        self.idx_list = idx_list
        self.edge_index_list = edge_index_list

        self.sa1_module = SAModule(in_channels, 32, 64, 3, conv_type=conv_type)
        self.sa2_module = SAModule(64+3, 64, 128, 3, conv_type=conv_type)

        self.bn_module = SAModule(128+3, 128, 256, conv_type=conv_type)

        self.gp1_module = GlobalSAModule(MLP([128 + 3, 512, 1024, global_token_dim]))

        self.fp2_module = FPModule(global_token_dim+256+64, 1024, 512, 3, conv_type=conv_type)
        self.fp1_module = FPModule(512 + in_channels, 256, 128, 3, conv_type=conv_type)

        self.mlp = MLP([128, 64, out_channels], norm=None)

    def forward(self, data):
        idx_list, edge_index_list = duplicate_edge_index(data, self.edge_index_list, self.idx_list)
    
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out, idx_list[0], edge_index_list[0], skip_pos_encoding=True)
        sa2_out = self.sa2_module(*sa1_out, idx_list[1], edge_index_list[1])
        
        bn_out = self.bn_module(*sa2_out, None, edge_index_list[2])

        sa2_out = self.gp1_module(*sa2_out, *bn_out)
        fp2_out = self.fp2_module(*sa2_out, *sa1_out, edge_index_list[1])
        fp1_out = self.fp1_module(*fp2_out, *sa0_out, edge_index_list[0])
        x, _, _ = fp1_out
        out = self.mlp(x)
        return out



def get_graph_conv(conv_type: str, in_channels: int, out_channels: int, **kwargs):
    conv_type = conv_type.lower()
    if conv_type == "gcn":
        return GCNConv(in_channels, out_channels)
    elif conv_type == "gat":
        return GATConv(in_channels, out_channels, heads=1, concat=False)
    elif conv_type == "cheb":
        k = kwargs.get("cheb_k", 3)
        return ChebConv(in_channels, out_channels, K=k)
    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")



class NavieUNet_V2(nn.Module):
    def __init__(self, in_channels, out_channels, idx_list, edge_index_list, global_token_dim=2048, conv_type='gcn', 
                 encn_list=[64, 128, 256], decn_list=[512, 128]):
        super().__init__()
        self.idx_list = idx_list
        self.edge_index_list = edge_index_list

        self.sa1_module = SAModule(in_channels, 32, encn_list[0], 3, conv_type=conv_type)
        self.sa2_module = SAModule(encn_list[0]+3, 64, encn_list[1], 3, conv_type=conv_type)

        self.bn_module = SAModule(encn_list[1]+3, 128, encn_list[2], conv_type=conv_type)

        self.gp1_module = GlobalSAModule(MLP([encn_list[1] + 3, 512, 1024, global_token_dim]))

        self.fp2_module = FPModule(global_token_dim+encn_list[2]+encn_list[0], 1024, decn_list[0], 3, conv_type=conv_type)
        self.fp1_module = FPModule(decn_list[0] + in_channels, 256, decn_list[1], 3, conv_type=conv_type)

        self.mlp = MLP([decn_list[1], 64, out_channels], norm=None)

    def forward(self, data):
        idx_list, edge_index_list = duplicate_edge_index(data, self.edge_index_list, self.idx_list)
    
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out, idx_list[0], edge_index_list[0], skip_pos_encoding=True)
        sa2_out = self.sa2_module(*sa1_out, idx_list[1], edge_index_list[1])
        
        bn_out = self.bn_module(*sa2_out, None, edge_index_list[2])

        sa2_out = self.gp1_module(*sa2_out, *bn_out)
        fp2_out = self.fp2_module(*sa2_out, *sa1_out, edge_index_list[1])
        fp1_out = self.fp1_module(*fp2_out, *sa0_out, edge_index_list[0])
        x, _, _ = fp1_out
        out = self.mlp(x)
        return out