#!/usr/bin/env python3
"""lod_coverage_fig.py — the LoD picture, as simple as it gets (Paul 2026-10-13: "show only training views … it doesn't show what we are trying
to do"): one row per orange (the largest oranges of a training view), columns = ground truth | render BEFORE | render AFTER | the gaussians
that sit on the orange BEFORE | AFTER, each drawn as a circle of its world size over the ground truth. What we are trying to do: the same
orange rendered by more, smaller gaussians, and therefore sharper. "On the orange" = centre projects inside the mask within 0.5 m of the
rendered fruit depth. Numbers per orange and run (count, median size cm, mask PSNR, alpha-weighted primitive size) -> <out>.json.
nerf_new pixi env:  lod_coverage_fig.py --before <config.yml> [--after <config.yml>] --sup <id maps> --frames-json <frames_fruit.json>
                    [--frame kf.png] [--n-oranges 3] [--pad 2.5] --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--before', required=True); ap.add_argument('--after', default=''); ap.add_argument('--sup', required=True)
ap.add_argument('--frames-json', default=''); ap.add_argument('--frame', default=''); ap.add_argument('--n-oranges', type=int, default=3); ap.add_argument('--pad', type=float, default=2.5)
ap.add_argument('--depth-band-m', type=float, default=0.5); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
runs = [('before', Path(a.before))] + ([('after', Path(a.after))] if a.after else [])
# ---- the training view with the most fruit pixels (or --frame) and its largest oranges ----
_, pipe, _, _ = eval_setup(runs[0][1], test_mode='test'); ds = pipe.datamanager.train_dataset; names = {Path(p).name: i for i, p in enumerate(ds.image_filenames)}
cand = [a.frame] if a.frame else json.load(open(a.frames_json))['frames']; best = None
for nm in cand:
    if nm not in names or not (Path(a.sup) / nm).exists(): continue
    ids = np.array(Image.open(Path(a.sup) / nm), np.uint16); n = int(((ids >= FRUIT) & (ids < UNLAB)).sum())
    if n and (best is None or n > best[0]): best = (n, nm)
assert best, 'no fruit frame is a training view with fruit ids in the supervision'
frame = best[1]; idx = names[frame]; cam0 = ds.cameras[idx:idx + 1]; H, W = int(cam0.height.item()), int(cam0.width.item())
ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / frame), np.uint16)).resize((W, H), Image.NEAREST)); mall = (ids >= FRUIT) & (ids < UNLAB)
lab, nlab = ndimage.label(mall); sizes = np.bincount(lab.ravel())[1:]; order = np.argsort(-sizes)[:a.n_oranges]
oranges = []
for k in order:
    m = lab == k + 1; ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    r = max(max(y1 - y0, x1 - x0) / 2 * a.pad, 24); Y0, Y1, X0, X1 = int(max(cy - r, 0)), int(min(cy + r, H)), int(max(cx - r, 0)), int(min(cx + r, W))
    oranges.append({'id': int(k + 1), 'px': int(sizes[k]), 'mask': m, 'crop': (Y0, Y1, X0, X1)})
gt = ds[idx]['image'][..., :3].cpu().numpy(); del pipe; torch.cuda.empty_cache()
print(f'[lod-fig] {frame} (training view, {nlab} oranges, {int(mall.sum())} fruit px): oranges ' + ', '.join(f"#{o['id']} {o['px']} px" for o in oranges), flush=True)
res = {}
for name, cfg in runs:
    _, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
    ds = pipe.datamanager.train_dataset; cam = ds.cameras[idx:idx + 1].to(dev); dps = float(json.load(open(cfg.parent / 'dataparser_transforms.json'))['scale'])
    g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); scales = torch.exp(g['scales'].detach()); size_m = scales.max(dim=-1).values / dps
    with torch.no_grad():
        viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev)
        feat, _, _ = rasterization(means=means, quats=g['quats'].detach(), scales=scales, opacities=torch.sigmoid(g['opacities'].detach()).squeeze(-1),
                                   colors=torch.stack([size_m, torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1), sh_degree=None, viewmats=viewmat, Ks=K,
                                   width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        out = model.get_outputs(cam); rgb = out['rgb']
        if getattr(model.config, 'use_bilateral_grid', False): rgb = model._apply_bilateral_grid(rgb[None], idx, H, W)[0]
        depth = out['depth'].squeeze(-1).cpu().numpy()
    psize_cm = (feat[0, ..., 0] / feat[0, ..., 1].clamp(min=1e-6) * 100).cpu().numpy(); rgb = rgb.clamp(0, 1).cpu().numpy()
    P = (viewmat[0] @ torch.cat([means, torch.ones(N, 1, device=dev)], 1).T).T; Z = P[:, 2]; front = Z > 1e-4
    u = (K[0, 0, 0] * P[:, 0] / Z.clamp(min=1e-4) + K[0, 0, 2]); v = (K[0, 1, 1] * P[:, 1] / Z.clamp(min=1e-4) + K[0, 1, 2]); rad_px = (K[0, 0, 0] * scales.max(dim=-1).values / Z.clamp(min=1e-4))
    ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1); per = []
    for o in oranges:
        m = o['mask']; dz = float(np.median(depth[m])); on = front & torch.from_numpy(m).to(dev)[vi, ui] & ((Z - dz).abs() < a.depth_band_m * dps)
        per.append({'n_on_orange': int(on.sum()), 'median_cm': float(size_m[on].median() * 100) if on.any() else float('nan'), 'primitive_size_cm_over_mask': float(np.nanmean(psize_cm[m])),
                    'psnr_mask_db': float(10 * np.log10(1 / max(((rgb - gt) ** 2)[m].mean(), 1e-10))), 'u': (u[on]).cpu().numpy().tolist(), 'v': (v[on]).cpu().numpy().tolist(), 'r': rad_px[on].cpu().numpy().tolist()})
    res[name] = {'gaussians': int(N), 'rgb': rgb, 'oranges': per, 'frame_psnr_fruit_db': float(10 * np.log10(1 / max(((rgb - gt) ** 2)[mall].mean(), 1e-10)))}
    print(f'[lod-fig] {name} ({cfg.parent.parent.name}, {N} gaussians): ' + '; '.join(f"orange #{o['id']}: {p['n_on_orange']} gaussians on it (median {p['median_cm']:.1f} cm), {p['psnr_mask_db']:.1f} dB" for o, p in zip(oranges, per)) + f"; frame fruit {res[name]['frame_psnr_fruit_db']:.1f} dB", flush=True)
    del pipe, model; torch.cuda.empty_cache()
ncol = 1 + 2 * len(runs); fig, axs = plt.subplots(len(oranges), ncol, figsize=(3.6 * ncol, 3.7 * len(oranges)), squeeze=False)
for r, o in enumerate(oranges):
    Y0, Y1, X0, X1 = o['crop']; cs = (slice(Y0, Y1), slice(X0, X1)); mc = o['mask'][cs]
    axs[r, 0].imshow(gt[cs], interpolation='nearest'); axs[r, 0].contour(mc, levels=[0.5], colors='white', linewidths=1.0); axs[r, 0].set_title(f'orange #{o["id"]}: ground truth', fontsize=11)
    for j, (name, _) in enumerate(runs):
        p = res[name]['oranges'][r]; ax = axs[r, 1 + j]; ax.imshow(res[name]['rgb'][cs], interpolation='nearest'); ax.contour(mc, levels=[0.5], colors='white', linewidths=1.0)
        ax.set_title(f'{name}: render  ({p["psnr_mask_db"]:.1f} dB on the orange)', fontsize=11)
        ax = axs[r, 1 + len(runs) + j]; ax.imshow(gt[cs], interpolation='nearest', alpha=0.45); ax.contour(mc, levels=[0.5], colors='white', linewidths=1.0)
        uu, vv, rr = np.array(p['u']) - X0, np.array(p['v']) - Y0, np.array(p['r']); o_ = np.argsort(-rr)
        ax.scatter(uu[o_], vv[o_], s=np.clip(np.pi * rr[o_] ** 2, 6, 6000), facecolors='#e8730c', edgecolors='#7a3b00', linewidths=0.8, alpha=0.35)
        ax.scatter(uu, vv, s=6, c='#7a3b00'); ax.set_xlim(0, X1 - X0); ax.set_ylim(Y1 - Y0, 0)
        ax.set_title(f'{name}: {p["n_on_orange"]} gaussians on the orange\nmedian size {p["median_cm"]:.1f} cm', fontsize=11)
    for c in range(ncol): axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
fig.suptitle(f'{frame} (training view): circles = gaussians whose centre lies on the orange, drawn at their world size', fontsize=12); fig.tight_layout()
fig.savefig(a.out, dpi=160)
json.dump({'frame': frame, 'oranges': [{k: v for k, v in o.items() if k != 'mask'} for o in oranges], 'runs': {n: {'gaussians': r_['gaussians'], 'frame_psnr_fruit_db': r_['frame_psnr_fruit_db'], 'oranges': [{k: v for k, v in p.items() if k not in ('u', 'v', 'r')} for p in r_['oranges']]} for n, r_ in res.items()}}, open(Path(a.out).with_suffix('.json'), 'w'), indent=1)
print(f'[lod-fig] -> {a.out}', flush=True)
