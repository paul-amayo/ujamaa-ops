#!/usr/bin/env python
"""langsplat_render_named.py — render a trained LangSplat baseline for NAMED cameras (paper E3, 2026-10-13): our verdict frames, whose
poses come from the chunk workspace's COLMAP model (or any extra --sparse dirs for frames outside it), at the scoring size. One npz per
frame: features [H,W,9] = the three levels' rendered 3-d language features side by side (as LangSplat renders them: per-gaussian
features unit-normalised, alpha-blended), rgb [H,W,3] uint8 from the RGB checkpoint, nodes = gaussians. langsplat env.
  langsplat_render_named.py --rgb-ckpt <chkpnt.pth> --feature-ckpts <ls_1 ckpt> <ls_2> <ls_3> --sparse <sparse/0> [...] --frames f.png ...
                            --out-dir <dir> [--width 640 --height 360]"""
import argparse, os, sys
sys.path.insert(0, '/home/paperspace/code/LangSplat'); os.chdir('/home/paperspace/code/LangSplat')
import numpy as np, torch
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
ap = argparse.ArgumentParser(); ap.add_argument('--rgb-ckpt', required=True); ap.add_argument('--feature-ckpts', nargs='+', required=True); ap.add_argument('--sparse', nargs='+', required=True)
ap.add_argument('--frames', nargs='+', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--width', type=int, default=640); ap.add_argument('--height', type=int, default=360)
a = ap.parse_args(); torch.set_grad_enabled(False); dev = 'cuda'
class Opt: pass
class Pipe: convert_SHs_python = False; compute_cov3D_python = False; debug = False
def load(ckpt):
    g = GaussianModel(3); mp, it = torch.load(ckpt, map_location=dev); o = Opt(); o.include_feature = True; g.restore(mp, o, mode='test'); return g
poses = {}
for sp in a.sparse:   # first model that knows the frame wins (the workspace, then the chunk export for off-workspace frames)
    ex = read_extrinsics_binary(os.path.join(sp, 'images.bin')); ins = read_intrinsics_binary(os.path.join(sp, 'cameras.bin'))
    for im in ex.values():
        n = os.path.basename(im.name)
        if n in poses: continue
        c = ins[im.camera_id]; fx, fy = (c.params[0], c.params[1]) if c.model in ('PINHOLE', 'OPENCV') else (c.params[0], c.params[0])
        poses[n] = (np.transpose(qvec2rotmat(im.qvec)), np.array(im.tvec), focal2fov(fx, c.width), focal2fov(fy, c.height))
rgb_model = load(a.rgb_ckpt); feat_models = [load(p) for p in a.feature_ckpts]; bg = torch.zeros(3, device=dev)
o_rgb = Opt(); o_rgb.include_feature = False; o_f = Opt(); o_f.include_feature = True; os.makedirs(a.out_dir, exist_ok=True)
print(f'[ls-render] {rgb_model.get_xyz.shape[0]} gaussians, {len(feat_models)} feature levels, {len(poses)} known cameras', flush=True)
for fr in a.frames:
    if fr not in poses: print(f'[ls-render] {fr}: no pose in {a.sparse}', flush=True); continue
    R, T, fovx, fovy = poses[fr]; cam = Camera(0, R, T, fovx, fovy, torch.zeros(3, a.height, a.width), None, fr, 0)
    rgb = render(cam, rgb_model, Pipe, bg, o_rgb)['render'].clamp(0, 1)
    feats = torch.cat([render(cam, m, Pipe, bg, o_f)['language_feature_image'] for m in feat_models], 0)   # [9,H,W]
    np.savez_compressed(os.path.join(a.out_dir, fr.replace('.png', '.npz')), features=feats.permute(1, 2, 0).float().cpu().numpy(), rgb=(rgb.permute(1, 2, 0) * 255).byte().cpu().numpy(), nodes=int(rgb_model.get_xyz.shape[0]))
    print(f'[ls-render] {fr}: {a.width}x{a.height}, level norms ' + ' / '.join(f'{feats[3 * l:3 * l + 3].norm(dim=0).mean().item():.2f}' for l in range(len(feat_models))), flush=True)
