import time
from pathlib import Path
import os
import json

import numpy as np
import torch
import viser
import viser.transforms as tf
from viser.theme import TitlebarButton, TitlebarConfig
from renderer import Renderer
from utils.cam_utils import *


class Viewer:
    def __init__(self, data_path: Path, args) -> None:
        self.data_path = data_path

        self.server = viser.ViserServer()
        self._setup_theme()
        self._setup_gui()
        
        self.required_update = False
        self.bg = torch.Tensor([0.,0.,0.]).cuda()
        self.frames = []

        # default rendering setting
        # TODO: need to be modified
        self.render_mode = "RGB"
        self.scale = 1.
        self.FoV = 72
        self.H = args.height
        self.W = args.width

        # load the output of 3DGS
        self.renderer = Renderer(data_path, fov=self.FoV, H=self.H, W=self.W)

        # lookat
        self.mean_pos = self.renderer.gs.get_mean_loc().detach().cpu().numpy()

    def _setup_theme(self) -> None:
        buttons = (
            TitlebarButton(
        text="Github",
        icon="GitHub",
        href="https://github.com/seung0h",),
        )

        title_theme = TitlebarConfig(buttons=buttons, image=None)

        self.server.configure_theme(titlebar_content=title_theme, show_logo=False, control_layout='floating', brand_color=(49, 96,190))

    def _setup_gui(self) -> None:
        self.gui_render_mode = self.server.add_gui_dropdown("Mode", ("RGB", "Depth"))
        self.gui_scaling = self.server.add_gui_slider("scaling factor", 0., 1., 0.1, 1.)
        self.gui_fov = self.server.add_gui_number("FoV", 72)

        @self.gui_render_mode.on_update
        def _(_) -> None:
            self.render_mode = self.gui_render_mode.value

        @self.gui_scaling.on_update
        def _(_) -> None:
            self.scale = self.gui_scaling.value

        @self.gui_fov.on_update
        def _(_) -> None:
            self.renderer.update_fov(self.gui_fov.value)

        # moving a viewpoint
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            # location initialization
            client.camera.look_at = self.mean_pos

            @client.camera.on_update
            def _(_: viser.CameraHandle):
                self.required_update = True

    def update_check(self):
        return self.required_update

    def update(self) -> None:
        '''
        send the rendered image to viser
        '''
        for cli in self.server.get_clients().values():
            cam = cli.camera
            w2c = get_w2c(cam)
            
            img = self.renderer.render(w2c, cam.position, self.bg, self.scale, self.render_mode).permute(1,2,0).cpu().detach().numpy()

            if img.shape[-1] == 1:
                img = np.repeat(img/img.max(), 3, axis=-1)
            
            cli.set_background_image(img, 'jpeg')
