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
from torch_geometric.nn import SplineConv, GCNConv, NNConv
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention
import math
from new_version.SplineCNNUNet import duplicate_edge_index

# NNConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, nn: Callable, aggr: str = 'add', root_weight: bool = True, bias: bool = True, **kwargs)


class ECGNSAModule(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 n_layers=3, patch_size=None, 
                 drop_act=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        assert n_layers >= 2, 'Number of layers should be at least 2'
        self.conv = torch.nn.ModuleList()
        self.drop_act = drop_act
        for i in range(n_layers):
            if i == 0:
                self.conv.append(NNConv(in_channels, hidden_channels, aggr='mean', nn=MLP([3, in_channels*hidden_channels], plain_last=False)))
            elif i == n_layers - 1:
                self.conv.append(NNConv(hidden_channels, out_channels, aggr='mean', nn=MLP([3, hidden_channels*out_channels], plain_last=False)))
            else:
                self.conv.append(NNConv(hidden_channels, hidden_channels, aggr='mean', nn=MLP([3, hidden_channels*hidden_channels], plain_last=False)))
        
    def forward(self, x, pos, batch, idx, edge_index):
        # compute edge length
        if self.patch_size is not None:
            pseudo = (pos[edge_index[1]] - pos[edge_index[0]]) / self.patch_size + 0.5
            pseudo = pseudo.clamp(min=0, max=1)
        else:
            pseudo = pos[edge_index[1]] - pos[edge_index[0]]
        # convolution
        for i in range(self.n_layers):
            if (i == self.n_layers - 1) and self.drop_act:
                x = self.conv[i](x, edge_index, pseudo)                
            else:
                x = F.relu(self.conv[i](x, edge_index, pseudo))
        # pooling
        if idx is not None:
            x, pos, batch = x[idx], pos[idx], batch[idx]
        return x, pos, batch
    

class ECGNFPModule(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 n_layers=3, patch_size=None, 
                 edge_dim=3, global_token_dim=1024,
                 embed_dim=64, num_heads=4,
                 drop_act=False, k=3):
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.n_layers = n_layers
        assert n_layers >= 2, 'Number of layers should be at least 2'
        self.conv = torch.nn.ModuleList()
        self.drop_act = drop_act
        for i in range(n_layers):
            if i == 0:
                self.conv.append(NNConv(in_channels, hidden_channels, aggr='mean', nn=MLP([edge_dim+embed_dim, in_channels*hidden_channels], plain_last=False)))
            elif i == n_layers - 1:
                self.conv.append(NNConv(hidden_channels, out_channels, aggr='mean', nn=MLP([edge_dim+embed_dim, hidden_channels*out_channels], plain_last=False)))
            else:
                self.conv.append(NNConv(hidden_channels, hidden_channels, aggr='mean', nn=MLP([edge_dim+embed_dim, hidden_channels*hidden_channels], plain_last=False)))

        self.W_q = nn.Linear(edge_dim, embed_dim)
        self.W_k = nn.Linear(global_token_dim, embed_dim)
        self.W_v = nn.Linear(global_token_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.embed_dim = embed_dim

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip, edge_index, global_token):
        # upsample
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        # compute edge length
        if self.patch_size is not None:
            pseudo = (pos_skip[edge_index[1]] - pos_skip[edge_index[0]]) / self.patch_size + 0.5
            pseudo = pseudo.clamp(min=0, max=1)
        else:
            pseudo = pos_skip[edge_index[1]] - pos_skip[edge_index[0]]
        # global token self-attention
        B = batch_skip.max().item() + 1
        edge_pos = 0.5 * (pos_skip[edge_index[1]] + pos_skip[edge_index[0]])
        edge_pos = edge_pos.view(B, -1, 3)
        global_token = global_token.view(B, -1, global_token.size(-1))
        Q = self.W_q(edge_pos)
        K = self.pos_encoder(self.W_k(global_token))
        V = self.pos_encoder(self.W_v(global_token))
        global_token, _ = self.attn(Q, K, V, need_weights=False)
        global_token = global_token.reshape(-1, self.embed_dim)
        # cat edge features
        pseudo = torch.cat([pseudo, global_token], dim=-1)
        # convolution
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        for i in range(self.n_layers):
            if (i == self.n_layers - 1) and self.drop_act:
                x = self.conv[i](x, edge_index, pseudo)                
            else:
                x = F.relu(self.conv[i](x, edge_index, pseudo))
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
        
class ECGN_V1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, idx_list, edge_index_list, patch_size_list, 
                 attn_embed_dim=64, attn_num_heads=8,
                 global_token_dim=1024, edge_dim=3,):
        super().__init__()
        self.num_layers = len(idx_list)
        self.idx_list = idx_list
        self.edge_index_list = edge_index_list
        self.patch_size_list = patch_size_list

        self.sa1_module = ECGNSAModule(in_channels, 16, 32, 2, patch_size_list[0])
        self.sa2_module = ECGNSAModule(32, 32, 64, 2, patch_size_list[1])

        self.bn_module = ECGNSAModule(64, 64, 64, 2, patch_size_list[2])

        self.gp1_module = GlobalSAModule(MLP([64 + 3, 256, global_token_dim]))

        self.fp2_module = ECGNFPModule(global_token_dim+64+32, 128, 64, 2, patch_size_list[1], edge_dim, global_token_dim+64, 
                                       embed_dim=attn_embed_dim, num_heads=attn_num_heads, k=3)
        self.fp1_module = ECGNFPModule(64+6, 32, 32, 2, patch_size_list[0], edge_dim, global_token_dim+64, 
                                       embed_dim=attn_embed_dim, num_heads=attn_num_heads, k=3)
        self.mlp = MLP([32, 16, out_channels], norm=None)

    def forward(self, data):
        idx_list, edge_index_list = duplicate_edge_index(data, self.edge_index_list, self.idx_list)
    
        sa0_out = (data.x[:, :6], data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out, idx_list[0], edge_index_list[0])
        sa2_out = self.sa2_module(*sa1_out, idx_list[1], edge_index_list[1])
        
        bn_out = self.bn_module(*sa2_out, None, edge_index_list[2])

        sa2_out = self.gp1_module(*sa2_out, *bn_out)
        fp2_out = self.fp2_module(*sa2_out, *sa1_out, edge_index_list[1], sa2_out[0])
        fp1_out = self.fp1_module(*fp2_out, *sa0_out, edge_index_list[0], sa2_out[0])
        x, _, _ = fp1_out
        out = self.mlp(x)
        return out


class PositionalEncoding(nn.Module):
    """
    see https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


