import pyvista as pv
import math
import trimesh
import numpy as np
import os
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError



pv.set_jupyter_backend('html')
pv.start_xvfb()


class VisualizationGeneral():
    def __init__(self, image_to_pyg_data_module, exponentialsmoothl2loss_module,
                 image_norm, labels, y_labels, faces, root):
        self.image_to_pyg_data_module = image_to_pyg_data_module
        self.exponentialsmoothl2loss_module = exponentialsmoothl2loss_module
        self.image_norm = image_norm
        self.labels = labels
        self.y_labels = y_labels
        self.faces = faces
        self.root = root


    def plot(self, data, model, pattern_only=True, add_global_mesh=True, plot_vectors=False, plot_parent_vessels=False, clim_max=25):
        batch, shape = self.image_to_pyg_data_module.forward(data['image'])
        case_name = data['case_name'][0]

        if add_global_mesh:
            global_mesh_path = os.path.join(self.root, case_name, "shape_remeshed.obj")
            global_mesh = pv.read(global_mesh_path)
            global_mesh.points *= 0.001
        else:
            global_mesh = None

        y_pred = model(batch).detach()
        y_true = batch.y.detach()

        true_original = self.exponentialsmoothl2loss_module.inspect_real(y_true, shape, self.image_norm, self.labels, self.y_labels)[0].cpu()
        pred_original = self.exponentialsmoothl2loss_module.inspect_real(y_pred, shape, self.image_norm, self.labels, self.y_labels)[0].cpu()
        true_original_norm = true_original.norm(dim=-1)
        pred_original_norm = pred_original.norm(dim=-1)
        true_mapped = self.exponentialsmoothl2loss_module.inspect(y_true, shape, pattern_only=pattern_only)[0].cpu()
        pred_mapped = self.exponentialsmoothl2loss_module.inspect(y_pred, shape, pattern_only=pattern_only)[0].cpu()
        true_mapped_norm = true_mapped.norm(dim=-1)
        pred_mapped_norm = pred_mapped.norm(dim=-1)

        rL2_original = (true_original - pred_original).norm() / true_original.norm()
        rL2_mapped = (true_mapped - pred_mapped).norm() / true_mapped.norm()
        # print(f"rL2_original: {rL2_original.item():.4f}, rL2_mapped: {rL2_mapped.item():.4f}")
        mse_original = nn.MSELoss()(true_original, pred_original)
        mse_mapped = nn.MSELoss()(true_mapped, pred_mapped)
        print(f"MSE_original: {rL2_original.item():.4f}, MSE_mapped: {rL2_mapped.item():.4f}")

        mae_metric = MeanAbsoluteError().to(y_true.device)
        rmse_metric = MeanSquaredError(squared=False).to(y_true.device)  # Set squared=False for RMSE
        mae_original = mae_metric(true_original, pred_original)
        rmse_original = rmse_metric(true_original, pred_original)
        mae_mapped = mae_metric(true_mapped, pred_mapped)
        rmse_mapped = rmse_metric(true_mapped, pred_mapped)
        print(f"MAE_original: {mae_original.item():.4f}, RMSE_original: {rmse_original.item():.4f}")
        print(f"MAE_mapped: {mae_mapped.item():.4f}, RMSE_mapped: {rmse_mapped.item():.4f}")


        if plot_vectors and true_original.shape[-1] == 3:
            true_original_vectors = true_original / true_original_norm.unsqueeze(-1)
            pred_original_vectors = pred_original / pred_original_norm.unsqueeze(-1)
            true_mapped_vectors = true_mapped / true_mapped_norm.unsqueeze(-1)
            pred_mapped_vectors = pred_mapped / pred_mapped_norm.unsqueeze(-1)
            print("true_original_vectors", true_original_vectors.shape)


        else:
            true_original_vectors = None
            pred_original_vectors = None
            true_mapped_vectors = None
            pred_mapped_vectors = None

        B = batch.batch.max() + 1
        x_true = batch.x[:, :3].view(B, -1, 3).detach()
        xyz = x_true[0].cpu()
        radius_mean = self.image_norm['mean'][..., 0]
        radius_std = self.image_norm['std'][..., 0]
        xyz = xyz * (radius_std.item() + 1e-5) + radius_mean.item()
        xyz = xyz.numpy()

        mesh = trimesh.Trimesh(vertices=xyz, faces=self.faces.cpu().numpy())

        vertices = mesh.vertices
        faces = mesh.faces

        # Convert faces to PyVista format: [3, i, j, k, 3, ...]
        faces_pv = np.hstack([[3] + list(face) for face in faces]).astype(np.int32)
        mesh_pv = pv.PolyData(vertices, faces_pv)

        label1 = "WSS (ground truth)"
        label2 = "WSS (prediction)"
        label3 = "Normalized WSS (ground truth)"
        label4 = "Normalized WSS (prediction)"

        mesh1 = mesh_pv.copy()
        mesh1[label1] = true_original_norm.numpy()
        mesh2 = mesh_pv.copy()
        mesh2[label2] = pred_original_norm.numpy()
        mesh3 = mesh_pv.copy()
        mesh3[label3] = true_mapped_norm.numpy()
        mesh4 = mesh_pv.copy()
        mesh4[label4] = pred_mapped_norm.numpy()

        plotter = pv.Plotter(shape=(2, 2), window_size=[1600, 1200])

        # Top-left
        plotter.subplot(0, 0)
        plotter.add_text(label1, font_size=12, position='upper_left', color='white', viewport=True)
        plotter.add_mesh(mesh1, scalars=label1, cmap="viridis", show_edges=False, clim=[0, clim_max])
        if plot_parent_vessels:
            plotter.add_mesh(global_mesh, color="white", show_edges=False, opacity=0.1)
        if true_original_vectors is not None:
            pdata = pv.PolyData(xyz)
            pdata["vectors"] = true_original_vectors.numpy()
            arrows = pdata.glyph(orient="vectors", scale=False, factor=radius_mean * 0.05)
            plotter.add_mesh(arrows, color="black")

        # Top-right
        plotter.subplot(0, 1)
        plotter.add_text(label2, font_size=10)
        plotter.add_mesh(mesh2, scalars=label2, cmap="viridis", show_edges=False, clim=[0, clim_max])
        if plot_parent_vessels:
            plotter.add_mesh(global_mesh, color="white", show_edges=False, opacity=0.1)
        if pred_original_vectors is not None:
            pdata = pv.PolyData(xyz)
            pdata["vectors"] = pred_original_vectors.numpy()
            arrows = pdata.glyph(orient="vectors", scale=False, factor=radius_mean * 0.05)
            plotter.add_mesh(arrows, color="black")

        # Bottom-left
        plotter.subplot(1, 0)
        plotter.add_text(label3, font_size=10)
        plotter.add_mesh(mesh3, scalars=label3, cmap="viridis", show_edges=False)
        if plot_parent_vessels:
            plotter.add_mesh(global_mesh, color="white", show_edges=False, opacity=0.1)

        # Bottom-right
        plotter.subplot(1, 1)
        plotter.add_text(label4, font_size=10)
        plotter.add_mesh(mesh4, scalars=label4, cmap="viridis", show_edges=False)
        if plot_parent_vessels:
            plotter.add_mesh(global_mesh, color="white", show_edges=False, opacity=0.1)

        plotter.link_views()  # optional: link camera views
        # plotter.show()



        return plotter