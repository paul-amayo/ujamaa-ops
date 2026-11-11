#!/usr/bin/env python3
"""twin_ledger_floor_look.py (11 Nov 2026, A100 session) - JIRS test JS-1: POST HOC, NOT A RULE. A look at what the twin renders
in the lane-floor band (2-4 m ahead, |x| <= 0.3 m) that gives the lane constants (December 0.383 m, April 1.570 m against the
lidar's 0.509 / 0.507 m). For one keyframe per twin (the panel frames of rank q=0.75): the splat's colour render, its expected
depth, and every pixel of the half-resolution render coloured by its height below the camera along the pass's levelling
direction (lane band outlined). nerf_new pixi env:  twin_ledger_floor_look.py <prep.json> <out dir>"""
import json, sys
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(3); torch.set_grad_enabled(False)
P = json.load(open(sys.argv[1])); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); GL = np.diag([1.0, -1.0, -1.0, 1.0])
for season, cell, kf in (('dec', 'lane10', 2076), ('apr', 'chunk_0_1', 1911)):
    S = P[season]; C = S['cells'][cell]; Rw = np.array(S['R']); up = np.array(S['up'])
    _, pipe, _, _ = eval_setup(Path(C['config']), test_mode='test'); model = pipe.model; model.eval(); dev = model.device
    g = model.gauss_params; means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
    dp = json.load(open(Path(C['config']).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale']); fx, fy, cx, cy = C['K']; W_, H_ = C['wh']
    fr = next(f for f in C['frames'] if f['k'] == kf); c2w = np.array(fr['c2w'], float); c = T @ (c2w @ GL); c[:3, 3] *= s
    cam = Cameras(camera_to_worlds=torch.tensor(c[:3], dtype=torch.float32)[None], fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(W_), height=int(H_), camera_type=CameraType.PERSPECTIVE).to(dev)
    cam.rescale_output_resolution(0.5); rgb = model.get_outputs(cam)['rgb'].clamp(0, 1).cpu().numpy(); viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
    fa, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=torch.ones((means.shape[0], 1), device=dev), viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB+ED', sh_degree=None, rasterize_mode=model.config.rasterize_mode)
    z = (fa[0, ..., 1] / s).cpu().numpy(); vv, uu = np.mgrid[0:h, 0:w]; Kn = K[0].cpu().numpy()
    pc = np.stack([(uu + 0.5 - Kn[0, 2]) / Kn[0, 0] * z, (vv + 0.5 - Kn[1, 2]) / Kn[1, 1] * z, z], -1); below = -((pc @ (Rw.T @ c2w[:3, :3]).T) @ up)   # camera height minus point height
    band = (z >= 2) & (z <= 4) & (np.abs(pc[..., 0]) <= 0.3)
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8)); ax[0].imshow(rgb); ax[0].set_title(f'{season} {cell} keyframe {kf}: splat colour render')
    im = ax[1].imshow(np.clip(z, 0, 8), cmap='viridis'); plt.colorbar(im, ax=ax[1], fraction=0.03); ax[1].set_title('expected depth (m, clipped at 8)')
    im = ax[2].imshow(np.where(z > 0, np.clip(below, -0.5, 2.5), np.nan), cmap='magma'); plt.colorbar(im, ax=ax[2], fraction=0.03); ax[2].contour(band, levels=[0.5], colors='cyan', linewidths=0.8)
    ax[2].set_title(f'camera height minus point height (m); lane band p90 {np.percentile(below[band], 90):.2f} m over {band.sum()} px')
    for a in ax: a.set_axis_off()
    fig.tight_layout(); fig.savefig(OUT / f'posthoc_floor_{season}.png', dpi=110); plt.close(fig)
    print(f'[floor] {season} {cell} k {kf}: band {band.sum()} px, p90 below-camera {np.percentile(below[band], 90):.3f} m, median {np.median(below[band]):.3f}; depth of the band px median {np.median(z[band]):.2f} m', flush=True)
    del model, pipe; torch.cuda.empty_cache()
