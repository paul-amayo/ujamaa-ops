#!/usr/bin/env python3
"""lift_pixel_features.py — LangSplat-lite, step 2 (2026-10-13): lift per-pixel feature maps into per-gaussian features with the alpha-blend
weights, closed form, no training — the census trick with continuous targets: rendering is linear in the colours, so the gradient of
sum(render * target) w.r.t. a zero colour tensor is each gaussian's alpha-weighted sum of the targets it blends into, and the gradient of
sum(render * covered) its total weight on covered pixels. Feature = sum / weight where the weight clears --floor (build_census_init's tree
floor 1.0), zero elsewhere. Writes <out> [N, K] float32 + json; render + score like any arm (argmax_eval.py --space flat --basis).
nerf_new pixi env:  lift_pixel_features.py --identity <ws>/identity --targets <dir of sam3_clip_targets.py> --out <bin> [--floor 1.0]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--targets', required=True, type=Path); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--floor', type=float, default=1.0)
a = ap.parse_args(); N = Path(a.identity); dev = 'cuda'; cfg = Path(json.load(open(N / 'W.json'))['config']); tj = json.load(open(a.targets / 'targets.json')); scale = tj['render_scale']
_, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); g = model.gauss_params; NG = g['means'].shape[0]
means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
ds = pipe.datamanager.train_dataset; cams = {Path(p).name: i for i, p in enumerate(ds.image_filenames)}
files = sorted(p for p in a.targets.glob('kf_*.npz')); K = tj['k']; num = torch.zeros(NG, K, device=dev); den = torch.zeros(NG, device=dev); t0 = time.time(); n = 0
torch.set_grad_enabled(True)
for p in files:
    f = p.name.replace('.npz', '.png')
    if f not in cams: continue
    z = np.load(p); tgt = torch.from_numpy(z['target']).to(dev); cov = torch.from_numpy(z['covered']).to(dev).float()
    cam = ds.cameras[cams[f]:cams[f] + 1].to(dev); cam.rescale_output_resolution(scale); h, w = tgt.shape[:2]
    col = torch.zeros(NG, K + 1, device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=get_viewmat(cam.camera_to_worlds), Ks=cam.get_intrinsics_matrices().to(dev), width=w, height=h,
                              sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gcol, = torch.autograd.grad((out[0, ..., :K] * tgt).sum() + (out[0, ..., K] * cov).sum(), col)
    num += gcol[:, :K]; den += gcol[:, K]; n += 1
    del out, col, gcol
sel = den > a.floor; F = torch.zeros(NG, K, device=dev); F[sel] = num[sel] / den[sel, None]
a.out.parent.mkdir(parents=True, exist_ok=True); F.cpu().numpy().astype(np.float32).tofile(a.out)
json.dump({'n': NG, 'dim': K, 'assigned': int(sel.sum()), 'frames': n, 'floor': a.floor, 'targets': str(a.targets), 'basis': str(a.targets / 'basis.npz'), 'seconds': time.time() - t0}, open(str(a.out) + '.json', 'w'), indent=1)
print(f'[lift] {N.parent.name}: {n} frames, {int(sel.sum())}/{NG} gaussians lifted ({100 * sel.float().mean().item():.1f}%), {time.time() - t0:.0f}s -> {a.out}', flush=True)
