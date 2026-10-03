"""Shared pieces of the native H3DGS x HiGH identity tools (UJAMAA, 2026-10-03): load a trained hierarchy the way
train_post / train_features.py do, index its cameras by image name, and expand one view to the render cut with the
render_post parent interpolation (the exact path RGB uses, so identity and colour come from the same gaussians).
Runs in the h3dgs conda env with CUDA_HOME=/home/paperspace/code/_cuda12 (gsplat's JIT backend needs nvcc)."""
import os, sys, math
H3 = '/home/paperspace/code/hierarchical-3d-gaussians'
for _p in (H3, os.path.join(H3, 'preprocess')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights


def add_scene_args(parser):
    from arguments import ModelParams, PipelineParams
    return ModelParams(parser), PipelineParams(parser)


def load(lp, args):
    from scene import Scene, GaussianModel
    from read_write_model import read_cameras_binary
    dataset = lp.extract(args)
    g = GaussianModel(dataset.sh_degree)
    g.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, g, resolution_scales=[1], create_from_hier=True)
    cam0 = list(read_cameras_binary(os.path.join(dataset.source_path, 'sparse', '0', 'cameras.bin')).values())[0]
    K0 = [float(x) for x in cam0.params[:4]]                       # fx, fy, cx, cy at the training resolution
    cams = {}
    for c in list(scene.getTrainCameras()) + list(scene.getTestCameras()):
        cams[c.image_name if c.image_name.endswith('.png') else c.image_name + '.png'] = c
    N = g._xyz.size(0)
    bufs = [torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(),
            torch.zeros(N).float().cuda(), torch.zeros(N).int().cuda()]
    return dataset, scene, g, cams, K0, bufs


def to_cuda(cam):
    for a in ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
        setattr(cam, a, getattr(cam, a).cuda())


def cut_threshold(cam, tau, cut_width):
    """render_hierarchy.py / hier_render_service.py: size threshold of the cut for a tau-pixel target at cut_width."""
    return (2 * (tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * cut_width)


def gsplat_cam(cam, K0, scale):
    viewmat = cam.world_view_transform.transpose(0, 1).contiguous().float().cuda()[None]   # H3DGS stores W2C transposed
    fx, fy, cx, cy = [v * scale for v in K0]
    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device='cuda')[None]
    return viewmat, Ks, int(round(cam.image_width * scale)), int(round(cam.image_height * scale))


def expand(g, cam, threshold, bufs):
    """train_post's per-view expansion + render_post's interpolation. Returns plain gaussians (no grad on geometry)
    and (ri, pi, w): rendered gaussian k = w_k * node[ri_k] + (1 - w_k) * node[pi_k]."""
    render_indices, parent_indices, nodes_for_render, interp_w, num_sib = bufs
    n = expand_to_size(g.nodes, g.boxes, threshold, cam.camera_center, torch.zeros((3)),
                       render_indices, parent_indices, nodes_for_render)
    node_idx = nodes_for_render[:n]
    get_interpolation_weights(node_idx, threshold, g.nodes, g.boxes, cam.camera_center.cpu(), torch.zeros((3)),
                              interp_w, num_sib)
    ri = render_indices[:n].long(); pi = parent_indices[:n].long(); w = interp_w[:n].unsqueeze(1); wi = 1 - w
    with torch.no_grad():
        xyz, sc, rot, op = g.get_xyz, g.get_scaling, g.get_rotation, g.get_opacity
        means = w * xyz[ri] + wi * xyz[pi]; scales = w * sc[ri] + wi * sc[pi]
        par = rot[pi].clone(); r = rot[ri]; par[(r * par).sum(1) < 0] *= -1; rots = w * r + wi * par
        opac = (w * op[ri] + wi * op[pi]).squeeze(-1)
    return means.contiguous(), scales.contiguous(), rots.contiguous(), opac.contiguous(), ri, pi, w


def with_skybox(g, means, scales, rots, opac):
    """Skybox gaussians (the last g.skybox_points nodes, outside the tree) occlude like in RGB; they carry no identity."""
    sky = int(getattr(g, 'skybox_points', 0) or 0)
    if not sky:
        return means, scales, rots, opac, 0
    N = g._xyz.size(0); sk = torch.arange(N - sky, N, device='cuda')
    with torch.no_grad():
        return (torch.cat([means, g.get_xyz[sk]]), torch.cat([scales, g.get_scaling[sk]]),
                torch.cat([rots, g.get_rotation[sk]]), torch.cat([opac, g.get_opacity[sk].squeeze(-1)]), sky)
