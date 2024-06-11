# Web-3DGS
<img src="figs/main.png" alt="image" width="70%" height="auto">

`Web-3DGS` is an interactive visualzation tool for 3d gaussian splatting based on the awesome library `Viser`. It would freely move anywhere by `W`,`A`,`S`,`D` or mouse movements, and it could be enabled to render RGB and Depth.


This repository is <b>working on progress</b> for the first release. There are two reasons that I start this project:
- First, the existing viewers are depdened on the physical monitor or complex framework, so it is tricky to modify and utilize it for my own projects. I would realize convenient open-sourced 3DGS viewer for easy adoptation.
- Second, the rendering type is also limited to only RGB. I just want to show various types of rendering such as depth and feature (like feature3DGS, LangSplat).


## Dependency
```
pytorch==1.13.1
viser==0.2.1
gsplat==1.0.0
ninja==1.11.1.1
```
- The version depends on GPU architecture, and I tested on RTX 4090 with cuda 11.6.

## How to use
```
python main.py -s {.ply_path_from_3dgs} 
```
