from typing import Dict
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import numpy as np
import pandas as pd



class RegisteredImageDataset(Dataset):
    def __init__(self, images, ghd_dict, image_norm, labels, dataset_labels, device=torch.device("cuda:0"), case_names=None):
        self.images = images
        self.ghd_dict = ghd_dict
        self.image_norm = image_norm
        self.labels = labels
        self.dataset_labels = dataset_labels
        self.channel_idx = [labels.index(label) for label in dataset_labels]
        self.case_names = case_names
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][..., self.channel_idx]
        ghd = self.ghd_dict['ghd'][idx]
        data = {'image': image, 'ghd': ghd}
        if self.case_names is not None:
            data['case_name'] = self.case_names[idx]
        return data


    





