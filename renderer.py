from pathlib import Path
import math

from plyfile import PlyData
import numpy as np
import torch
import torch.nn as nn

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from model import Gaussians
from utils.sh_utils import eval_sh
from utils.cam_utils import *


deg2rad = lambda x: x*(math.pi / 180)

class Renderer:
    def __init__(self, data_path: Path, 
                 fov=100, H=800, W=800, z_near=0.1, z_far=100) -> None:
        self.data_path = data_path
        
        self.model = Gaussians()
        self.model.load_ply(data_path)

        self.active_sh_degree = self.model.get_max_sh_degress

        self.FoVx = deg2rad(fov)
        self.FoVy = deg2rad(fov)
        self.image_height = H
        self.image_width = W

        self.zear = z_near
        self.zfar = z_far

        self.proj = getProjectionMatrix(self.zear, self.zfar, self.FoVx, self.FoVy).transpose(0,1).cuda()

    
    def update_fov(self, deg):
        rad = deg2rad(deg)

        self.FoVx = rad
        self.FoVy = rad

        self.proj = getProjectionMatrix(self.zear, self.zfar, self.FoVx, self.FoVy).transpose(0,1).cuda()

    def render(self, R, T, bg_color : torch.Tensor, scaling_modifier = 0.5, mode='RGB', override_color = None):
        """
        Render an image
        """
        # temp: arguments
        compute_cov3D_python = False
        convert_SHs_python = False

        world_view_transform = torch.tensor(getWorld2View2(R, T, scale=scaling_modifier)).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.proj.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.model.get_xyz, dtype=self.model.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(self.FoVx * 0.5)
        tanfovy = math.tan(self.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.model.get_xyz
        means2D = screenspace_points
        opacity = self.model.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.model.get_scaling
            rotations = self.model.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.model.get_features.transpose(1, 2).view(-1, 3, (self.active_sh_degree+1)**2)
                dir_pp = (self.model.get_xyz - camera_center.repeat(self.model.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.model.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        if mode == 'Depth':            
            return rendered_depth
        return rendered_image

    def reload(self, iters=30000) -> None:
        self.model.load_ply(self.data_path, iters)
