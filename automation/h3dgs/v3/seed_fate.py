#!/usr/bin/env python3
"""seed_fate.py — what became of the seeded fruit blobs (Paul 2026-10-14: "what exactly happens to it at 15 or 30k, is it moved away?"):
for every seed in the run's fruit_seeds.npz (position, radius, step), find the blob nearest its seed position in a later checkpoint; it 'survived'
if one lies within its radius. Report survival, how far the survivor moved, its size vs the held cap, its opacity, and its labelled fruit share from
the checkpoint's census W (unlabelled px do not vote). Figure on --frame (same crop as the other figures, the n largest oranges): photo with the
seeds as drawn at seeding (dashed circles at their world size) | the checkpoint's render with the survivors (solid, filled when >= 50 % fruit by
votes, grey when lost) — so moved / shrunk / lost is visible per orange.
nerf_new pixi env:  seed_fate.py --config <config_stepN.yml> --seeds <fruit_seeds.npz> --w-npz <W.npz of that checkpoint> --sup <id maps>
                    --frame kf.png [--n-oranges 3] --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--seeds', required=True); ap.add_argument('--w-npz', default=''); ap.add_argument('--sup', required=True)
ap.add_argument('--frame', required=True); ap.add_argument('--n-oranges', type=int, default=3); ap.add_argument('--pad', type=float, default=2.5); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
S = np.load(a.seeds); P0 = torch.from_numpy(S['pts']).float(); R0 = torch.from_numpy(S['r']).float(); st0 = S['step']
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; ds = pipe.datamanager.train_dataset
dps = float(json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))['scale']); g = model.gauss_params; N = g['means'].shape[0]
means = g['means'].detach(); size = torch.exp(g['scales'].detach()).max(dim=-1).values; opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
P0, R0 = P0.to(dev), R0.to(dev); d = torch.cdist(P0, means); dmin, j = d.min(1); surv = dmin <= R0
f = None
if a.w_npz:
    Z = np.load(a.w_npz); W, lab = Z['W'], Z['labels'].tolist(); fr = [i for i, l in enumerate(lab) if FRUIT <= l < UNLAB]; labd = [i for i, l in enumerate(lab) if l != UNLAB]
    fm = W[fr].sum(0); lm = W[labd].sum(0); f = torch.from_numpy(np.where(lm > 0, fm / np.maximum(lm, 1e-9), 0.0).astype(np.float32)).to(dev)
mv_cm = (dmin / dps * 100); sz_cm = size[j] / dps * 100; r_cm = R0 / dps * 100
res = {'seeds': int(P0.shape[0]), 'survived': int(surv.sum()), 'median_move_cm_survivors': float(mv_cm[surv].median()) if surv.any() else float('nan'), 'median_size_cm_survivors': float(sz_cm[surv].median()) if surv.any() else float('nan'),
       'median_seed_radius_cm': float(r_cm.median()), 'median_opacity_survivors': float(opac[j][surv].median()) if surv.any() else float('nan'),
       'survivors_fruit_majority': int(((f[j] >= 0.5) & surv).sum()) if f is not None else None, 'median_fruit_share_survivors': float(f[j][surv].median()) if (f is not None and surv.any()) else None,
       'by_seed_step': {int(s): {'seeds': int((st0 == s).sum()), 'survived': int(surv[torch.from_numpy(st0 == s).to(dev)].sum())} for s in np.unique(st0)}}
print(f"[fate] step {step}: {res['seeds']} seeds, {res['survived']} have a blob within their radius ({100 * res['survived'] / max(res['seeds'], 1):.0f} %; by seed step {res['by_seed_step']}); survivors moved median {res['median_move_cm_survivors']:.1f} cm, "
      f"size median {res['median_size_cm_survivors']:.1f} cm (seed radius {res['median_seed_radius_cm']:.1f} cm), opacity median {res['median_opacity_survivors']:.2f}"
      + (f", fruit-majority by votes {res['survivors_fruit_majority']} of {res['survived']} (median share {res['median_fruit_share_survivors']:.2f})" if f is not None else ''), flush=True)
# ---- figure on the frame ----
names = [Path(p).name for p in ds.image_filenames]; idx = names.index(a.frame); cam = ds.cameras[idx:idx + 1].to(dev); H, Wd = int(cam.height.item()), int(cam.width.item())
with torch.no_grad():
    out = model.get_outputs(cam); rgb = out['rgb']
    if getattr(model.config, 'use_bilateral_grid', False): rgb = model._apply_bilateral_grid(rgb[None], idx, H, Wd)[0]
    rgb = rgb.clamp(0, 1).cpu().numpy()
gt = ds[idx]['image'][..., :3].cpu().numpy(); ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / a.frame), np.uint16)).resize((Wd, H), Image.NEAREST)); mall = (ids >= FRUIT) & (ids < UNLAB)
lab_, n = ndimage.label(mall); sizes = np.bincount(lab_.ravel())[1:]; order = np.argsort(-sizes)[:a.n_oranges]
vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev)
def proj(X, rad):
    Pc = (vm[0] @ torch.cat([X, torch.ones(X.shape[0], 1, device=dev)], 1).T).T; z = Pc[:, 2].clamp(min=1e-4)
    return (K[0, 0, 0] * Pc[:, 0] / z + K[0, 0, 2]).cpu().numpy(), (K[0, 1, 1] * Pc[:, 1] / z + K[0, 1, 2]).cpu().numpy(), (K[0, 0, 0] * rad / z).cpu().numpy(), (Pc[:, 2] > 0).cpu().numpy()
u0, v0, r0, in0 = proj(P0, R0); u1, v1, r1, in1 = proj(means[j], size[j])
fig, axs = plt.subplots(len(order), 2, figsize=(8.2, 4.0 * len(order)), squeeze=False); nshown = 0
for row, k in enumerate(order):
    m = lab_ == k + 1; ys, xs = np.where(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; r = max(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * a.pad, 24)
    Y0, Y1, X0, X1 = int(max(cy - r, 0)), int(min(cy + r, H)), int(max(cx - r, 0)), int(min(cx + r, Wd)); cs = (slice(Y0, Y1), slice(X0, X1))
    sel = in0 & (u0 >= X0) & (u0 < X1) & (v0 >= Y0) & (v0 < Y1); ii = np.where(sel)[0]; nshown += len(ii)
    axs[row, 0].imshow(gt[cs], interpolation='nearest'); axs[row, 0].contour(m[cs], levels=[0.5], colors='white', linewidths=1.2)
    for i in ii: axs[row, 0].add_patch(plt.Circle((u0[i] - X0, v0[i] - Y0), max(r0[i], 1.5), fill=False, ls='--', color='#e8730c', lw=1.6))
    axs[row, 0].set_title(f'orange #{k + 1}: photo + {len(ii)} seeds as placed (step {", ".join(str(int(s)) for s in np.unique(st0[ii]))})' if len(ii) else f'orange #{k + 1}: photo (no seed in crop)', fontsize=9.5)
    axs[row, 1].imshow(rgb[cs], interpolation='nearest'); axs[row, 1].contour(m[cs], levels=[0.5], colors='white', linewidths=1.2); ns, nf = 0, 0
    for i in ii:
        if bool(surv[i]):
            pure = f is not None and float(f[j[i]]) >= 0.5; ns += 1; nf += int(pure)
            axs[row, 1].add_patch(plt.Circle((u1[i] - X0, v1[i] - Y0), max(r1[i], 1.5), fill=pure, alpha=0.5 if pure else 1.0, color='#e8730c', lw=1.6))
            axs[row, 1].plot([u0[i] - X0, u1[i] - X0], [v0[i] - Y0, v1[i] - Y0], color='#e8730c', lw=0.8)
        else:
            axs[row, 1].plot(u0[i] - X0, v0[i] - Y0, 'x', color='#888888', ms=9, mew=2)
    axs[row, 1].set_title(f'step {step}: {ns} of {len(ii)} seeds survive (line = move), {nf} are ≥ 50 % fruit by votes (filled); × = lost', fontsize=9.5)
    for c in range(2): axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
fig.suptitle(f'{a.frame}: fate of the seeded fruit blobs at step {step} — {res["survived"]} of {res["seeds"]} survive, moved {res["median_move_cm_survivors"]:.1f} cm (median), size {res["median_size_cm_survivors"]:.1f} cm', fontsize=10.5)
fig.tight_layout(); fig.savefig(a.out, dpi=150); json.dump(res, open(Path(a.out).with_suffix('.json'), 'w'), indent=1); print(f'[fate] -> {a.out}', flush=True)
