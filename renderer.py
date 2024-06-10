from pathlib import Path
import math
import torch
from gsplat import spherical_harmonics, rasterize_gaussians, project_gaussians
from model import Gaussians
from utils.cam_utils import *


deg2rad = lambda x: x*(math.pi / 180)

class Renderer:
    def __init__(self, data_path: Path, 
                 fov=100, H=800, W=800, z_near=0.1, z_far=100) -> None:
        self.data_path = data_path
        self.B_SIZE = 16
        
        self.gs = Gaussians(data_path)

        self.active_sh_degree = self.gs.get_max_sh_degress

        self.image_height = H
        self.image_width = W

        self.zear = z_near
        self.zfar = z_far

        self.update_fov(fov)

    def update_fov(self, deg):
        rad = deg2rad(deg)

        self.FoVx = rad
        self.FoVy = 2 * math.atan(self.image_height / (2 * (self.image_width / (2 * math.tan(self.FoVx / 2)))))

        self.proj = getProjectionMatrix(self.zear, self.zfar, self.FoVx, self.FoVy).transpose(0,1).cuda()

    def render(self, w2c, center, bg_color : torch.Tensor, scaling_modifier = 0.5, mode='RGB'):
        center = torch.tensor(center).float().cuda()
        w2c = torch.tensor(w2c).float().cuda()

        (
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
        cov3d,
        ) = project_gaussians(
            self.gs.get_xyz,
            self.gs.get_scaling,
            1,
            self.gs.get_rotation,
            w2c,
            1468.2172,
            1468.2172,
            self.image_width / 2 - 0.5,
            self.image_height / 2 - 0.5,
            self.image_height,
            self.image_width,
            self.B_SIZE,
        )
        
        dir_pp = (self.gs.get_xyz - center.repeat(self.gs.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = spherical_harmonics(self.active_sh_degree, dir_pp_normalized, self.gs.get_features)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        torch.cuda.synchronize()
        out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                colors_precomp,
                self.gs.get_opacity,
                self.image_height,
                self.image_width,
                self.B_SIZE,
                bg_color,
            )[..., :3]
        out_img = out_img.permute(2,0,1)
        
        torch.cuda.synchronize()

        return out_img
