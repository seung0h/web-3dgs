from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData


class Gaussians:
    def __init__(self) -> None:
        self._xyz = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self._feature_dc = None
        self._feature_rest = None

        self.max_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)
    
    @property
    def get_scaling(self):
        return torch.exp(self._scaling)
    
    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)
    
    @property
    def get_features(self):
        return self._features
    
    @property
    def get_max_sh_degress(self):
        return self.max_sh_degree

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def load_ply(self, data_path: Path, iter_num=30000) -> None:
        plydata = PlyData.read(data_path / f"point_cloud/iteration_{str(iter_num)}/point_cloud.ply")
        gs_data = plydata.elements[0]

        xyz = np.stack((np.asarray(gs_data["x"]),
                        np.asarray(gs_data["y"]),
                        np.asarray(gs_data["z"])),  axis=1)
        opacities = np.asarray(gs_data["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(gs_data["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(gs_data["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(gs_data["f_dc_2"])

        extra_f_names = [p.name for p in gs_data.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(gs_data[attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scales = np.zeros((xyz.shape[0], 3))
        scales[:, 0] = np.asarray(gs_data["scale_0"])
        scales[:, 1] = np.asarray(gs_data["scale_1"])
        scales[:, 2] = np.asarray(gs_data["scale_2"])

        rots = np.zeros((xyz.shape[0], 4))
        rots[:, 0] = np.asarray(gs_data["rot_0"])
        rots[:, 1] = np.asarray(gs_data["rot_1"])
        rots[:, 2] = np.asarray(gs_data["rot_2"])
        rots[:, 3] = np.asarray(gs_data["rot_3"])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features = torch.cat((
            nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)), 
            nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            ), dim=1)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
