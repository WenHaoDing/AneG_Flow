import torch
from torch.utils.data import Dataset, DataLoader
try:
    import dgl
    from dgl.data import DGLDataset
    from pytorch3d.transforms import so3_exp_map
except ImportError:
    print("dgl needs to be installed")
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader
from models.pointnet.pointnet2_utils import regroup_points, reweight_points
from typing import List
import copy
from torch_geometric.nn.unpool import knn_interpolate



"""
dgl datasets for wss segmentation
"""

class STEADY_MCA_ANEURYSM_DATASET_WSS_ADVANCED(Dataset):
    def __init__(self,
                 wss_data_path,
                 np_nodes,  # number of nodes per graph
                 np_knn=8,  # number of knn neighbor
                 input_keys=None,
                 output_keys=None,
                 pouch_only=True,
                 group_dict: Dict = None,
                 capping_threshold=7,
                 frame_to_extract=50
                 ):
        """
        TODO: from scipy.spatial.transform import Rotation as R
        """
        self.wss_pcd = torch.load(wss_data_path)
        if 'caseID' not in self.wss_pcd.keys():
            self.wss_pcd['caseID'] = torch.arange(self.wss_pcd['x'].shape[0]).unsqueeze(-1).unsqueeze(-1).expand(-1, self.wss_pcd[
                'x'].shape[1], 1)


        element = self.wss_pcd[input_keys[0]]
        self.shape_num, self.num_frames, self.np_total = element.shape[0], element.shape[1], element.shape[2]
        self.frame_to_extract = frame_to_extract

        # pouch only
        self.pouch_only = pouch_only
        if self.pouch_only:
            # self.relative_indices_pouch_list = self.wss_pcd['relative_pouch_indices']
            self.relative_indices_pouch_list = list(self.wss_pcd['relative_pouch_indices'].values())

        # keys
        self.input_keys = input_keys
        self.output_keys = output_keys

        # graph conf
        self.np_nodes = np_nodes
        self.np_knn = np_knn

        # grouping conf
        for key in group_dict:
            setattr(self, key, group_dict[key])
        self.new_xyz_list = []
        self.group_idx_list = []
        self.itp_weight_list = []
        self.itp_idx_list = []
        # graphs
        self.graphs = []  # List[List[]]

        # generate indices
        self.resample()
        # self.reconstruct_graph()

        # calclate osi and tawss
        if "osi" in self.output_keys or "tawss" in self.output_keys:
            advanced_scalars = self.get_advanced_scalars()
            for key_ in advanced_scalars:
                self.wss_pcd[key_] = advanced_scalars[key_]

        # torch.save(self.wss_pcd, "debug_osi.pth")
        # breakpoint()

        # wss capping
        self.capping_threshold = capping_threshold
        self.wss_capping()
        # normalize dataset
        self.normalize()

        # get input and output
        self.input_ = self.assemble_key_values(input_keys, self.wss_pcd)  # flow data input
        self.output_ = self.assemble_key_values(output_keys, self.wss_pcd)

    def normalize(self):
        # normalization
        self.norm_dict = {}
        self.xyz_std = None
        self.xyz_mean = None
        for key_ in [key__ for key__ in self.wss_pcd.keys() if
                     key__ not in ["x", "y", "z", "caseID", "relative_pouch_indices", "osi"] and isinstance(self.wss_pcd[key__], torch.Tensor)]:
            self.norm_dict["std_" + key_] = torch.std(self.wss_pcd[key_]) + 1e-2
            self.norm_dict["mean_" + key_] = torch.mean(self.wss_pcd[key_])
            self.wss_pcd[key_] = (self.wss_pcd[key_] - self.norm_dict["mean_" + key_]) / \
                                           self.norm_dict["std_" + key_]
        whole_r = self.return_whole_xyz().permute((0, 2, 1)).reshape(-1, 3).norm(dim=1)  # [B, 3, N] -> [-1, 3] -> [1]
        self.xyz_std = torch.std(whole_r)
        self.xyz_mean = torch.mean(whole_r)
        for key_ in ["x", "y", "z"]:
            self.wss_pcd[key_] = (self.wss_pcd[key_] - self.xyz_mean) / self.xyz_std

    def augmentation_init(self):
        # get pca axis
        self.pca_vectors = []
        whole_xyz = self.return_whole_xyz().permute(0, 2, 1)  # [B, N, 3]
        pca = torch.pca_lowrank(whole_xyz, niter=4, q=1)
        self.pca_vectors = pca[-1]  # [B, 3, 1]
        self.pca_vectors /= torch.norm(self.pca_vectors, dim=1, keepdim=True)


    def reconstruct_graph(self, to_device='cuda:0', graph_bt_size=1):
        """
        resample graph points -> calcualte grouping info -> create knn graphs and return
        """
        # resample points
        self.resample()
        whole_xyz = self.return_whole_xyz()  # [B, C, N]

        # regroup points
        self.regroup_n_reweight(whole_xyz, to_device=to_device)

        # construct graph
        self.graphs = []
        # self.graphs.append(dgl.unbatch(dgl.knn_graph(whole_xyz.permute(0, 2, 1), self.np_knn)))
        # for r_level in range(len(self.S_list)):
        #     self.graphs.append(dgl.unbatch(dgl.knn_graph(self.new_xyz_list[r_level].permute(0, 2, 1), self.np_knn)))
        # breakpoint()
        pt_tensors = [whole_xyz.to(torch.device(to_device))] + copy.deepcopy(self.new_xyz_list)
        dataset = CustomPointDataset(pt_tensors)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=graph_bt_size)
        for i, pt_list_ in enumerate(dataloader):
            if i == 0:
                for j in range(len(pt_list_)):
                    self.graphs.append(dgl.unbatch(dgl.knn_graph(pt_list_[j].permute(0, 2, 1), self.np_knn).add_self_loop()))
            else:
                for j in range(len(pt_list_)):
                    self.graphs[j] = self.graphs[j] + dgl.unbatch(dgl.knn_graph(pt_list_[j].permute(0, 2, 1), self.np_knn).add_self_loop())
        return self.new_xyz_list, self.group_idx_list, self.itp_weight_list, self.itp_idx_list, self.graphs

    def resample(self):  # resample interior points
        if not self.pouch_only:
            self.p_indices = torch.randperm(self.np_total)[:self.np_nodes].long()
        else:
            self.p_indices = []
            for i in range(len(self.relative_indices_pouch_list)):
                rand_i = torch.randint(0, len(self.relative_indices_pouch_list[i]), (self.np_nodes,))
                self.p_indices.append(self.relative_indices_pouch_list[i][rand_i])
            self.p_indices = torch.stack(self.p_indices, dim=0).long()  # [B, N]

    def return_whole_xyz(self):
        """
        return coordinates for all points & shape for pointnet2 grouping & weighting
        """
        whole_xyz = self.assemble_key_values(["x", "y", "z"], self.wss_pcd)

        if not self.pouch_only:
            whole_xyz = whole_xyz[:, :, self.p_indices]
        else:
            whole_xyz = torch.gather(whole_xyz, dim=2,
                                     index=self.p_indices.unsqueeze(1).expand(-1, whole_xyz.shape[1], -1))  # [B, 3, N]
        return whole_xyz

    def regroup_n_reweight(self, xyz, to_device, batch_size=10):
        """
        xyz: [B_all, 3, N] (pcd coordinates for ALL shapes)
        regroup and reweight using level 0 point coordinates
        use this function to calculate the information for ALL shapes
        """
        # device = xyz.get_device()
        device = torch.device(to_device)
        if device == -1:
            device = torch.device('cpu')
        xyz = xyz.detach().cpu()

        # grouping & weighting info for branch one
        self.new_xyz_list = []
        self.group_idx_list = []
        self.itp_weight_list = []
        self.itp_idx_list = []

        # assign workers
        wrapper = []
        dataset = TensorDataset(xyz)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for _, xyz_ in enumerate(dataloader):
            xyz_ = xyz_[0]
            filler = {'new_xyz_list': [],
                      'group_idx_list': [],
                      'itp_weight_list': [],
                      'itp_idx_list': []}
            xyz_l0 = xyz_  # level 0 points input for reweight
            level_num = len(self.S_list)
            for level in range(level_num):
                xyz_, group_idx_list_ = regroup_points(xyz_, self.S_list[level], self.radius_list[level],
                                                       self.K_list[level])
                filler['new_xyz_list'].append(xyz_)
                filler['group_idx_list'].append([ele for ele in group_idx_list_])
            print("pcd have been regrouped, info stored at new_xyz_list and group_idx_list")
            for r_level in range(level_num):
                if r_level != level_num - 1:
                    weight, weight_idx = reweight_points(filler['new_xyz_list'][level_num - r_level - 2],
                                                         filler['new_xyz_list'][level_num - r_level - 1])
                else:
                    weight, weight_idx = reweight_points(xyz_l0, filler['new_xyz_list'][
                        0])  # for the zero level use input xyz_l0 as interpolation target
                filler['itp_weight_list'].append(weight)
                filler['itp_idx_list'].append(weight_idx)
            wrapper.append(filler)
        # cat
        do1 = [f['new_xyz_list'] for f in wrapper]
        do2 = [f['itp_weight_list'] for f in wrapper]
        do3 = [f['itp_idx_list'] for f in wrapper]
        for i in range(len(self.S_list)):
            self.new_xyz_list.append(torch.cat([ele[i] for ele in do1], dim=0).to(device))
            self.itp_weight_list.append(torch.cat([ele[i] for ele in do2], dim=0).to(device))
            self.itp_idx_list.append(torch.cat([ele[i] for ele in do3], dim=0).to(device))
            do4 = [f['group_idx_list'][i] for f in wrapper]
            do4_ = []
            for j in range(len(self.radius_list[0])):
                do4_.append(torch.cat([ele[j] for ele in do4], dim=0).to(device))
            self.group_idx_list.append(do4_)
        print("interpolation weighting has been redone, info stored at itp_weight_list and itp_idx_list")

    def assemble_key_values(self, keys, tensor_dict):
        # default point tensor shape: [B, Nt, N, 1]
        # -> [B, C, N]
        tensors = []
        for key in keys:
            tensors.append(tensor_dict[key][:, self.frame_to_extract, ...] if tensor_dict[key].dim() == 4 else tensor_dict[key])
        # tensors = [tensor_dict[key][:, self.frame_to_extract, ...] for key in keys]
        y = torch.cat(tensors, dim=-1).permute(0, 2, 1)
        return y

    @staticmethod
    def assemble_key_values_all_frames(keys, tensor_dict):
        # default point tensor shape: [B, Nt, N, 1]
        # -> [B, Nt, N, C]
        tensors = [tensor_dict[key] for key in keys]
        y = torch.cat(tensors, dim=-1)
        return y

    def __getitem__(self, index):
        # default point tensor shape [B, C, N]
        if not self.pouch_only:
            data_dict = {"input": self.input_[index, :, self.p_indices],
                         "output": self.output_[index, :, self.p_indices],
                         "caseID": index}
        else:
            data_dict = {"input": self.input_[index, :, self.p_indices[index, :]],
                         "output": self.output_[index, :, self.p_indices[index, :]],
                         "caseID": index}
        multi_graphs = [graphs[index] for graphs in self.graphs]
        # graph = self.graphs[0][index]
        # graph.ndata['x'] = torch.zeros((graph.num_nodes(), 1))
        # graph.edata['x'] = torch.zeros((graph.num_edges(), 1))
        # breakpoint()
        return multi_graphs, data_dict

    def __len__(self):
        return self.shape_num

    def wss_capping(self):
        wss = self.assemble_key_values(self.output_keys, self.wss_pcd).permute(0, 2, 1)
        wss_mag = torch.norm(wss, dim=-1, keepdim=False)
        mask = (wss_mag >= self.capping_threshold)
        mask = (self.capping_threshold / wss_mag - 1) * mask + 1
        for key_ in ["wss_x", "wss_y", "wss_z"]:
            self.wss_pcd[key_][:, self.frame_to_extract, ...] = self.wss_pcd[key_][:, self.frame_to_extract, ...] * mask.unsqueeze(-1)
        print("global wss has been capped to {}".format(self.capping_threshold))
        if ["tawss"] in self.output_keys:
            tawss = self.assemble_key_values(["tawss"], self.wss_pcd).permute(0, 2, 1)
            mask = (tawss >= self.capping_threshold)
            mask = (self.capping_threshold / tawss - 1) * mask + 1
            self.wss_pcd["tawss"] = self.wss_pcd["tawss"] * mask.unsqueeze(-1)
            print("tawss has been capped to {}".format(self.capping_threshold))

    def get_advanced_scalars(self):
        wss = self.assemble_key_values_all_frames(["wss_x", "wss_y", "wss_z"], self.wss_pcd)  # [B, Nt, N, 3]
        wss_magnitude = torch.norm(wss, dim=-1, keepdim=True)  # [B, Nt, N, 1]
        tawss = torch.mean(wss_magnitude, dim=1, keepdim=False)  # [B, N, 1]
        numerator = torch.norm(torch.mean(wss, dim=1, keepdim=False), dim=-1, keepdim=True)  # [B, N, 1]
        osi = 0.5*(1-numerator/tawss)
        advanced_scalars = {"tawss": tawss,
                            "osi": osi}
        return advanced_scalars


class CustomPointDataset(Dataset):
    def __init__(self, tensors: List):
        self.tensors = tensors
        self.num_layer = len(self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        pt_list = [pts[idx, ...] for pts in self.tensors]
        return pt_list


def rodrigues_rotation(theta: torch.Tensor, coords: torch.Tensor, pca_vectors: torch.Tensor) -> torch.Tensor:
    """
    theta: [B, 1]
    coords: [B, N, 3]
    pca_vectors: [B, 3, 1]
    pytorch3d.transforms.so3_exp_map
    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
    """
    if pca_vectors.dim() == 3:
        pca_vectors = pca_vectors.squeeze(-1)
    log_rot = theta * pca_vectors
    rotational_matrices = so3_exp_map(log_rot)  # [B, 3, 3]
    rotated_coords = torch.einsum('bij,bnj->bni', rotational_matrices, coords)
    return rotated_coords


def norm_vectors(input_: torch.Tensor):
    """
    input_: [N, 3]

    """
    norm = torch.norm(input_, dim=1, keepdim=True)
    input_ = input_ / norm
    return input_


"""
https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html#sphx-glr-tutorials-blitz-5-graph-classification-py



"""
