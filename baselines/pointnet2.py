# %%
import numpy as np
from new_version.loaders import read_fluent_ascii, extract_last_fluent_residuals
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from new_version.loaders import get_fluent_ascii_raw_data
import torch
from new_version.loaders import register_raw_data_to_image_4k, get_trimmed_indice
from AneuG.models.ghd_reconstruct import GHD_Reconstruct
from AneuG.utils.utils import safe_load_mesh
import pickle
from new_version.datasets_wss import RegisteredImageDataset

root = "/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/solutions/stable_64_v3"
write_path = os.path.join(root, "registered_data_1k_clean_full.pth")
normalize_image = True
n_subdivide = 1
downsampling_factors = [4, 4]
data_object = register_raw_data_to_image_4k(None, None, None, write_path=write_path, 
                                            overwrite=False, normalize=None, 
                                            n_subdivide=None, mask=None, insert_normals=None, downsampling_factors=None)
images = data_object['images']
ghd_dict = data_object['ghd_dict']
image_norm = data_object['image_norm']
labels = data_object['labels']
idx_list = data_object['idx_list']
edge_index_list = data_object['edge_index_list']
downsampling_factors = data_object['downsampling_factors']
print("downsampling_factors: ", downsampling_factors)
# %%
device = torch.device("cuda:0")
batch_size = 155
dataset_labels = ["x", "y", "z", "x_normal", "y_normal", "z_normal", "wss_x", "wss_y", "wss_z"]
dataset = RegisteredImageDataset(images, ghd_dict, image_norm, labels, dataset_labels, device=device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(24)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

# %%
from new_version.models import SimplePointNet2, ImageToPYGData, SimplePointNet2_V2, SimplePointNet2_V3
from new_version.losses import ExponentialSmoothL2Loss
x_labels = ["x", "y", "z", "x_normal", "y_normal", "z_normal"]
y_labels = ["wss_x", "wss_y", "wss_z"]
# model = SimplePointNet2(num_classes=len(y_labels))
model = SimplePointNet2_V2(num_classes=len(y_labels))
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
mseloss_module = torch.nn.MSELoss()
data_range = (images.max(dim=1, keepdim=True)[0] - images.min(dim=1, keepdim=True)[0])[:, :, [labels.index(label_) for label_ in y_labels]].mean(dim=0, keepdim=True)
print(data_range.shape, data_range)
exponentialsmoothl2loss_module = ExponentialSmoothL2Loss(n=2, range_max=100, data_range=data_range, device=device)
image_to_pyg_data_module = ImageToPYGData(dataset_labels, x_labels=x_labels, y_labels=y_labels, device=device)

# %%
import wandb
meta = "Pointnet2_v2_1k_vectors_stable"
log_path = os.path.join(root, meta, "log")
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_wandb = True
if log_wandb:
    config = {
        "version": 2,
        "batch_size": batch_size,
        "lr": 5e-4,
        "write_path": write_path,
        "x_labels": x_labels,
        "y_labels": y_labels,
    }
    run = wandb.init(project="RHSIA_ModeNet", name=meta)
# %%
reload_epoch = None
if reload_epoch is not None:
    reload_path = os.path.join(log_path, f"model_epoch_{reload_epoch}.pth")
    chk = torch.load(reload_path)
    epoch_0 = chk["epoch"] if "epoch" in chk else 0
    model.load_state_dict(chk['model'])
    optimizer.load_state_dict(chk["optimizer"]) if "optimizer" in chk else None
    scheduler.load_state_dict(chk["scheduler"]) if "scheduler" in chk else None
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    print("checkpoints reloaded from {}".format(reload_path))
    test_mse_loss_ = 0
    test_esl2_loss_ = 0
    test_esl2_po_loss_ = 0
else:
    epoch_0 = 0
# %%

for epoch in range(epoch_0, 501):
    model.train()
    mse_loss_ = 0
    esl2_loss_ = 0
    esl2_po_loss_ = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = {key: value.to(device) for key, value in data.items()}
        batch, shape = image_to_pyg_data_module.forward(data['image'])
        y_pred = model(batch)
        y_true = batch.y
        mse_loss = mseloss_module(y_pred, y_true)
        esl2_loss = exponentialsmoothl2loss_module(y_pred, y_true, shape)
        esl2_po_loss = exponentialsmoothl2loss_module(y_pred, y_true, shape, pattern_only=True)
        loss = 1.0 * mse_loss + 50.0 * esl2_loss + 50.0 * esl2_po_loss
        loss = mse_loss
        loss.backward()
        optimizer.step()
        mse_loss_ += mse_loss.item()/len(train_loader)
        esl2_loss_ += esl2_loss.item()/len(train_loader)
        esl2_po_loss_ += esl2_po_loss.item()/len(train_loader)

    if epoch % 1 == 0:
        model.eval()
        test_mse_loss_ = 0
        test_esl2_loss_ = 0
        test_esl2_po_loss_ = 0
        data = next(iter(test_loader))
        data = {key: value.to(device) for key, value in data.items()}
        batch, shape = image_to_pyg_data_module.forward(data['image'])
        y_pred = model(batch)
        y_true = batch.y
        test_mse_loss = mseloss_module(y_pred, y_true)
        test_esl2_loss = exponentialsmoothl2loss_module(y_pred, y_true, shape)
        test_esl2_po_loss = exponentialsmoothl2loss_module(y_pred, y_true, shape, pattern_only=True)
        test_mse_loss_ += test_mse_loss.item()
        test_esl2_loss_ += test_esl2_loss.item()
        test_esl2_po_loss_ += test_esl2_po_loss.item()
    loss_log = {"train_mse_loss": mse_loss_, "train_esl2_loss": esl2_loss_, "train_esl2_po_loss": esl2_po_loss_,
                "test_mse_loss": test_mse_loss_, "test_esl2_loss": test_esl2_loss_, "test_esl2_po_loss": test_esl2_po_loss_
                }
    if log_wandb:
        wandb.log(loss_log, step=epoch)
    print(f"Epoch {epoch}: ", loss_log)
    if epoch % 100 == 0:
        # torch.save(model.state_dict(), os.path.join(log_path, f"model_epoch_{epoch}.pth"))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }, os.path.join(log_path, f"model_epoch_{epoch}.pth"))
        print(f"Model saved at epoch {epoch}")
    scheduler.step()
wandb.finish()

"""
cd /media/yaplab/HDD_Storage/wenhao/RHSIA
conda activate new
python pointnet2.py

"""