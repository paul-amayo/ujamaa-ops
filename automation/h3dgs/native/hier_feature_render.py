#!/usr/bin/env python3
"""Render per-node HiGH features (features.bin, float32 [N_nodes, 32]) through an H3DGS hierarchy's own cut
(UJAMAA, 2026-10-03). Same expansion + parent interpolation as RGB (hier_common.expand), gsplat 32-channel raster.
Writes one npz per (frame, tau): features [H,W,32] float32, alpha [H,W] float32, rgb [H,W,3] uint8 (SH render of the
same cut, for figure backdrops), nodes = gaussians in the cut — the layout containment_eval.py --features-npz reads.
h3dgs env (CUDA_HOME=/home/paperspace/code/_cuda12):  python hier_feature_render.py <scene args> --features <bin>
   --frames kf_a.png kf_b.png [--taus 3] [--cut-width 1280] [--render-scale 0.5] --out-dir <dir>"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_common as hc
import numpy as np
import torch
from gsplat import rasterization

ap = argparse.ArgumentParser()
lp, pp = hc.add_scene_args(ap)
ap.add_argument('--features', required=True)
ap.add_argument('--frames', nargs='+', required=True)
ap.add_argument('--taus', type=float, nargs='+', default=[3.0])
ap.add_argument('--cut-width', type=int, default=1280)
ap.add_argument('--render-scale', type=float, default=0.5)
ap.add_argument('--out-dir', required=True)
args = ap.parse_args()
dataset, scene, g, cams, K0, bufs = hc.load(lp, args)
N = g._xyz.size(0)
F = torch.from_numpy(np.fromfile(args.features, dtype=np.float32).reshape(N, -1)).cuda()
os.makedirs(args.out_dir, exist_ok=True)
print(f'[render] {N} nodes, features {tuple(F.shape)} ({(F.norm(dim=1) > 0).float().mean().item() * 100:.1f}% of nodes non-zero)', flush=True)
torch.set_grad_enabled(False)
for name in args.frames:
    if name not in cams:
        print(f'[render] {name}: not a camera of this hierarchy — skipped', flush=True); continue
    cam = cams[name]; hc.to_cuda(cam)
    viewmat, Ks, Wr, Hr = hc.gsplat_cam(cam, K0, args.render_scale)
    for tau in args.taus:
        t0 = time.time()
        means, scales, rots, opac, ri, pi, w = hc.expand(g, cam, hc.cut_threshold(cam, tau, args.cut_width), bufs)
        f = w * F[ri] + (1 - w) * F[pi]
        sh = g.get_features; shs = w.unsqueeze(2) * sh[ri] + (1 - w).unsqueeze(2) * sh[pi]
        means, scales, rots, opac, sky = hc.with_skybox(g, means, scales, rots, opac)
        if sky:
            sk = torch.arange(N - sky, N, device='cuda')
            f = torch.cat([f, torch.zeros(sky, F.shape[1], device='cuda')]); shs = torch.cat([shs, sh[sk]])
        kw = dict(means=means, quats=rots, scales=scales, opacities=opac, viewmats=viewmat, Ks=Ks, width=Wr, height=Hr,
                  packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode='classic')
        feat, alpha, _ = rasterization(colors=f.contiguous(), sh_degree=None, **kw)
        rgb, _, _ = rasterization(colors=shs.contiguous(), sh_degree=g.active_sh_degree, **kw)
        out = os.path.join(args.out_dir, f'{name[:-4]}_tau{tau:g}.npz')
        np.savez_compressed(out, features=feat[0].float().cpu().numpy(), alpha=alpha[0, ..., 0].float().cpu().numpy(),
                            rgb=(rgb[0].clamp(0, 1) * 255).byte().cpu().numpy(), nodes=int(ri.numel()))
        fn = feat[0].norm(dim=-1)
        print(f'[render] {name} tau {tau:g}: {ri.numel()} gaussians in the cut, {Wr}x{Hr}, ||f||>0.5 on '
              f'{(fn > 0.5).float().mean().item() * 100:.1f}% of px, {time.time() - t0:.1f}s -> {out}', flush=True)
