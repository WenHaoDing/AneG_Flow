import torch
import torch.nn as nn
import math
from torch_geometric.data import Data, Batch


class ExponentialSmoothL2Loss(nn.Module):
    def __init__(self, n=3, range_max=100, data_range=None, device=None):
        super().__init__()
        self.n = n
        self.range_max = range_max
        self.data_range = data_range.to(device) if data_range is not None else None
        self.loss_module = torch.nn.MSELoss()

    def map(self, x, pattern_only=False):
        eps = 1e-4  # small constant to avoid divide-by-zero or negative roots

        if pattern_only:
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            scale = torch.clamp(x_max - x_min, min=eps)
            normalized = (x - x_min) / scale
        else:
            x_min = x.min(dim=1, keepdim=True)[0]
            scale = torch.clamp(self.data_range, min=eps)
            normalized = (x - x_min) / scale

        # Clamp normalized range before root
        normalized = torch.clamp(normalized * self.range_max, min=eps)
        return torch.pow(normalized, 1 / self.n) / math.sqrt(self.range_max)

    def forward(self, pred, true, shape, pattern_only=False):
        B, N, _ = shape
        # pred = torch.stack(pred.to_data_list())
        # true = torch.stack(true.to_data_list())
        pred = pred.view(B, N, -1)
        true = true.view(B, N, -1)
        pred = self.map(pred, pattern_only=pattern_only)
        true = self.map(true, pattern_only=pattern_only)
        loss = self.loss_module(pred, true)
        return loss

    def inspect(self, true, shape, pattern_only=True):
        B, N, _ = shape
        # pred = torch.stack(pred.to_data_list())
        # true = torch.stack(true.to_data_list())
        true = true.view(B, N, -1)
        true = self.map(true, pattern_only=pattern_only)
        return true

    def inspect_real(self, true, shape, image_norm, labels, y_labels):
        B, N, _ = shape
        # pred = torch.stack(pred.to_data_list())
        # true = torch.stack(true.to_data_list())
        true = true.view(B, N, -1)
        idx = [labels.index(label) for label in y_labels]
        mean, std = image_norm['mean'][..., idx], image_norm['std'][..., idx]
        mean, std = mean.to(true.device), std.to(true.device)
        true = true * std + mean
        return true
