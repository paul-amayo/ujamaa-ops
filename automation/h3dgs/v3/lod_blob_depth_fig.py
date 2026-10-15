#!/usr/bin/env python3
"""lod_blob_depth_fig.py — the painters of ONE orange in the reference frame, same crop as lod_coverage_fig / lod_identity_fig (2026-10-14, Paul:
"keep it consistent, show me kf 25"): photo | the run's render | rendered depth (m), each with the orange's painters (from lod_trace_fig's json)
drawn at their projected size — orange = fruit word, green = tree word, grey = other / none; the main painter (largest mass share) outlined thick.
Title per painter: its own depth minus the rendered surface depth at its pixel (+ = behind the surface the camera sees there).
nerf_new pixi env:  lod_blob_depth_fig.py --config <config.yml> --trace-json <lod_trace_*.json> --sup <id maps> --frame kf.png --orange-rank 3 --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--trace-json', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--frame', required=True)
ap.add_argument('--orange-rank', type=int, default=3); ap.add_argument('--label', default='', help='name used in the earlier figures, e.g. 20'); ap.add_argument('--pad', type=float, default=2.5); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; ds = pipe.datamanager.train_dataset
names = [Path(p).name for p in ds.image_filenames]; idx = names.index(a.frame); cam = ds.cameras[idx:idx + 1].to(dev); H, W = int(cam.height.item()), int(cam.width.item())
dps = float(json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))['scale']); g = model.gauss_params; means = g['means'].detach(); scales = torch.exp(g['scales'].detach())
LBL = a.label or str(a.orange_rank)
ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / a.frame), np.uint16)).resize((W, H), Image.NEAREST)); mall = (ids >= FRUIT) & (ids < UNLAB)
lab, n = ndimage.label(mall); sizes = np.bincount(lab.ravel())[1:]; k = np.argsort(-sizes)[a.orange_rank - 1]; m = lab == k + 1
ys, xs = np.where(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; r = max(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * a.pad, 24)
Y0, Y1, X0, X1 = int(max(cy - r, 0)), int(min(cy + r, H)), int(max(cx - r, 0)), int(min(cx + r, W)); cs = (slice(Y0, Y1), slice(X0, X1))
with torch.no_grad():
    out = model.get_outputs(cam); rgb = out['rgb']
    if getattr(model.config, 'use_bilateral_grid', False): rgb = model._apply_bilateral_grid(rgb[None], idx, H, W)[0]
    rgb = rgb.clamp(0, 1).cpu().numpy(); depth = out['depth'].squeeze(-1).cpu().numpy() / dps
gt = ds[idx]['image'][..., :3].cpu().numpy(); T = json.load(open(a.trace_json)); P = T['painters']
viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); gi = torch.tensor([p['g'] for p in P], device=dev)
Pc = (viewmat[0] @ torch.cat([means[gi], torch.ones(len(gi), 1, device=dev)], 1).T).T; z = Pc[:, 2].clamp(min=1e-4)
u = (K[0, 0, 0] * Pc[:, 0] / z + K[0, 0, 2]).cpu().numpy(); v = (K[0, 1, 1] * Pc[:, 1] / z + K[0, 1, 2]).cpu().numpy(); rp = (K[0, 0, 0] * scales[gi].max(dim=-1).values / z).cpu().numpy(); zm = (z / dps).cpu().numpy()
surf = np.array([depth[int(np.clip(vv, 0, H - 1)), int(np.clip(uu, 0, W - 1))] for uu, vv in zip(u, v)]); main = int(np.argmax([p['mass_share'] for p in P]))
COL = {'fruit': '#e8730c', 'tree': '#2e8b3d', 'other-fruit': '#b05cff', 'none': '#888888'}
print(f'[blob] {a.frame} orange #{LBL}: orange surface {float(np.median(depth[m])):.2f} m; ' + '; '.join(f"g{p['g']} {p['word']} {p['size_cm']:.1f} cm {100 * p['mass_share']:.0f}%: depth {zz:.2f} m, surface {ss:.2f} m, behind by {zz - ss:+.2f} m" for p, zz, ss in zip(P, zm, surf)), flush=True)
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.6)); dm = depth[cs]
for ax, img, title in ((axs[0], gt[cs], 'photo'), (axs[1], rgb[cs], 'render')):
    ax.imshow(img, interpolation='nearest'); ax.contour(m[cs], levels=[0.5], colors='white', linewidths=1.2); ax.set_title(f'orange #{LBL}: {title}', fontsize=11)
im = axs[2].imshow(np.clip(dm, np.percentile(dm, 2), np.percentile(dm, 98)), cmap='Blues', interpolation='nearest'); axs[2].contour(m[cs], levels=[0.5], colors='black', linewidths=1.2); plt.colorbar(im, ax=axs[2], fraction=0.046, label='rendered depth (m)')
zo = float(np.median(depth[m])); axs[2].set_title(f"rendered depth: the orange at {zo:.2f} m\nmain blob g{P[main]['g']} at {zm[main]:.2f} m = {zm[main] - zo:+.2f} m behind the orange\n(its centre falls on a leaf at {surf[main]:.2f} m)", fontsize=9.5)
for ax in axs:
    o_ = np.argsort(-rp)
    for j in o_:
        ax.add_patch(plt.Circle((u[j] - X0, v[j] - Y0), max(rp[j], 1.5), fill=False, color=COL[P[j]['word']], lw=3.0 if j == main else 1.2, alpha=0.95 if j == main else 0.7))
    ax.scatter(u - X0, v - Y0, s=8, c='black'); ax.set_xlim(0, X1 - X0); ax.set_ylim(Y1 - Y0, 0); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'{a.frame}: the {len(P)} gaussians painting orange #{LBL} (orange = fruit word, green = tree word; thick = main painter g{P[main]["g"]}, {P[main]["size_cm"]:.1f} cm, {100 * P[main]["mass_share"]:.0f}% of its mass)', fontsize=10.5)
fig.tight_layout(); fig.savefig(a.out, dpi=150); print(f'[blob] -> {a.out}', flush=True)
