# %%
import numpy as np
from new_version.loaders import read_fluent_ascii, extract_last_fluent_residuals
from torch.utils.data import Dataset, DataLoader
import os
from new_version.loaders import get_fluent_ascii_raw_data
import torch
from new_version.loaders import register_raw_data_to_image_4k, get_trimmed_indice

root = "/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/solutions/stable_64_v3"
labels = ["x-coordinate", "y-coordinate", "z-coordinate", "wall-shear", "x-wall-shear", "y-wall-shear", "z-wall-shear"]
dataset_labels = ["x", "y", "z", "wss", "wss_x", "wss_y", "wss_z"]
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
from new_version.datasets_wss import RegisteredImageDataset

device = torch.device("cuda:0")
batch_size = 38
dataset_labels = ["x", "y", "z", "x_normal", "y_normal", "z_normal", "wss_x", "wss_y", "wss_z"]
dataset = RegisteredImageDataset(images, ghd_dict, image_norm, labels, dataset_labels, device=device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(24)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

# %%
from new_version.models import SimplePointNet2, ImageToPYGData
from new_version.UniversalNaiveUNets import NavieUNet_V1, NavieUNet_V2
from new_version.losses import ExponentialSmoothL2Loss
x_labels = ["x", "y", "z", "x_normal", "y_normal", "z_normal"]
y_labels = ["wss_x", "wss_y", "wss_z"]
kernel_size = 6
degree = 2
patch_size_list = [0.15, 0.35, 0.95]
# model = SimplePointNet2(num_classes=len(y_labels))
# model = SplineCNNUNet_V1(6, len(y_labels), idx_list, edge_index_list, patch_size_list, 
#                          kernel_size=kernel_size, degree=degree)
conv_type = "cheb"
# model = NavieUNet_V1(6, len(y_labels), idx_list, edge_index_list, conv_type=conv_type)
encn_list=[16, 32, 64]
decn_list=[64, 32]
global_token_dim = 1024
model = NavieUNet_V2(6, len(y_labels), idx_list, edge_index_list, conv_type=conv_type, global_token_dim=global_token_dim,
                     encn_list=encn_list, decn_list=decn_list)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
mseloss_module = torch.nn.MSELoss()
data_range = (images.max(dim=1, keepdim=True)[0] - images.min(dim=1, keepdim=True)[0])[:, :, [labels.index(label_) for label_ in y_labels]].mean(dim=0, keepdim=True)
print(data_range.shape, data_range)
exponentialsmoothl2loss_module = ExponentialSmoothL2Loss(n=2, range_max=100, data_range=data_range, device=device)
image_to_pyg_data_module = ImageToPYGData(dataset_labels, x_labels=x_labels, y_labels=y_labels, device=device)
# %%
import wandb
# meta = "SplineCNNUNet_V1_1k"
meta = "UniversalNaiveUNet_V1_1k_"+conv_type
log_path = os.path.join(root, meta, "log")
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_wandb = True
if log_wandb:
    config = {
        "conv_type": conv_type,
        "batch_size": batch_size,
        "lr": 5e-4,
        "write_path": write_path,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "encn_list": encn_list,
        "decn_list": decn_list,
        "global_token_dim": global_token_dim,
    }
    run = wandb.init(project="RHSIA_ModeNet", name=meta, config=config)

# %%
reload_epoch = None
if reload_epoch is not None:
    reload_path = os.path.join(log_path, f"model_epoch_{reload_epoch}.pth")
    chk = torch.load(reload_path)
    epoch_0 = chk["epoch"]+1 if "epoch" in chk else 0
    model.load_state_dict(chk['model'])
    optimizer.load_state_dict(chk["optimizer"]) if "optimizer" in chk else None
    scheduler.load_state_dict(chk["scheduler"]) if "scheduler" in chk else None
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
    print("checkpoints reloaded from {}".format(reload_path))
    test_mse_loss_ = 0
    test_esl2_loss_ = 0
    test_esl2_po_loss_ = 0
else:
    epoch_0 = 0

# %%
for epoch in range(epoch_0, int(501)):
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
        loss = 1 * mse_loss + 50.0 * esl2_loss + 50.0 * esl2_po_loss
        loss.backward()
        optimizer.step()
        mse_loss_ += mse_loss.item()/len(train_loader)
        esl2_loss_ += esl2_loss.item()/len(train_loader)
        esl2_po_loss_ += esl2_po_loss.item()/len(train_loader)
        # print(mse_loss.item(), esl2_loss.item(), esl2_po_loss.item())

    if epoch % 5 == 0:
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
python UniversalNaiveUNets.py

conda remove --name new --all --yes

conda create -n new python=3.11 --yes

conda activate new

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia --yes 
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

conda install -c conda-forge yacs --yes 
conda install -c iopath iopath --yes 
conda install pytorch3d -c pytorch3d --yes 
conda install anaconda::pandas --yes 
conda install conda-forge::trimesh --yes 
conda install conda-forge::matplotlib --yes 
conda install conda-forge::point_cloud_utils --yes 
conda install conda-forge::vtk --yes 
conda install conda-forge::igraph --yes 
conda install conda-forge::pyvista --yes 
conda install conda-forge::wandb --yes

pip install open3d
pip install git+https://github.com/igraph/python-igraph
pip3 install skeletor

python temp.py
python UNets.py


conda activate new


"""


