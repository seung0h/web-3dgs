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
        self.FoV = 50
        self.H = args.height
        self.W = args.width

        # load the output of 3DGS
        self.renderer = Renderer(data_path, fov=self.FoV, H=self.H, W=self.W)

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
        self.gui_fov = self.server.add_gui_number("FoV", 50)
        self.gui_iter_num = self.server.add_gui_dropdown("iteration number", load_iternums(self.data_path), "30000", visible=False)

        @self.gui_render_mode.on_update
        def _(_) -> None:
            self.render_mode = self.gui_render_mode.value

        @self.gui_scaling.on_update
        def _(_) -> None:
            self.scale = self.gui_scaling.value

        @self.gui_fov.on_update
        def _(_) -> None:
            self.renderer.update_fov(self.gui_fov.value)

        @self.gui_iter_num.on_update
        def _(_) -> None:
            self.renderer.reload(int(self.gui_iter_num.value))

        # moving a viewpoint
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            self.load_frames(client)

            @client.camera.on_update
            def _(_: viser.CameraHandle):
                self.required_update = True
    
    def add_frame(self, idx, wxyz, position, client):
        frame = self.server.add_frame(f"/frames/frame_{idx}", wxyz=wxyz, position=position)
        self.frames.append(frame)
        self.server.add_label(f"/frames/frame_{idx}/label", text=f"Frame {idx}")

        @frame.on_click
        def _(_) -> None:
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(20):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 19.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                client.flush()
                time.sleep(1.0 / 60.0)

            client.camera.look_at = frame.position

    def load_frames(self, client:viser.ClientHandle):
        with open(os.path.join(self.data_path, 'cameras.json'), 'r') as f:
            contents = json.load(f)

        for idx, item in enumerate(contents):
            position = np.array(item['position'])
            rotation = np.array(item['rotation']).reshape((3,3))

            q = tf.SO3.from_matrix(rotation)

            if idx == 0:
                client.camera.position = position
                client.camera.wxyz = q.wxyz

            # self.add_frame(idx, q.wxyz, position, client)

    def update_check(self):
        return self.required_update

    def update(self) -> None:
        '''
        send the rendered image to viser
        '''
        for cli in self.server.get_clients().values():
            cam = cli.camera
            w2c = get_w2c(cam)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3,-1]
            
            img = self.renderer.render(R, T, self.bg, self.scale, self.render_mode).permute(1,2,0).cpu().detach().numpy()

            if img.shape[-1] == 1:
                img = np.repeat(img/img.max(), 3, axis=-1)
            
            cli.set_background_image(img, 'jpeg')


def load_iternums(data_path) -> tuple:
    res = []
    for f in os.listdir(data_path / "point_cloud"):
        res.append(f.split("_")[-1])
    
    return res
