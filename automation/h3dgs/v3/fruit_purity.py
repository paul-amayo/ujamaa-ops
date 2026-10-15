#!/usr/bin/env python3
"""fruit_purity.py — the identity-LoD metric (2026-10-14): from a checkpoint's census W (labelled votes only — the unlabelled row does not
count), each blob's fruit share f = fruit mass / labelled mass; then on the given frames (fruit-picked training views) rasterise the indicator
f >= --pure, so per pixel we get the share of the rendered mass carried by PURE-fruit blobs, and report it averaged over the fruit pixels (and
per orange), plus the blob counts. A perfectly unmixed splat scores 1.0 on the fruit pixels.
nerf_new pixi env:  fruit_purity.py --config <config.yml> --w-npz <W.npz> --sup <id maps> --frames f.png ... [--pure 0.5] [--out json]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--w-npz', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--frames', nargs='+', required=True)
ap.add_argument('--pure', type=float, default=0.5); ap.add_argument('--min-lab', type=float, default=1.0); ap.add_argument('--out', default=''); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
Z = np.load(a.w_npz); W, lab = Z['W'], Z['labels'].tolist(); fr = [i for i, l in enumerate(lab) if FRUIT <= l < UNLAB]; labd = [i for i, l in enumerate(lab) if l != UNLAB]
fm = W[fr].sum(0); lm = W[labd].sum(0); f = np.where(lm > 0, fm / np.maximum(lm, 1e-9), 0.0); judged = lm >= a.min_lab
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; ds = pipe.datamanager.train_dataset
names = {Path(p).name: i for i, p in enumerate(ds.image_filenames)}; g = model.gauss_params; N = g['means'].shape[0]; assert W.shape[1] == N, f'W {W.shape[1]} vs checkpoint {N}'
pure = torch.from_numpy((judged & (f >= a.pure)).astype(np.float32)).to(dev); mixed = torch.from_numpy((judged & (f >= 0.1) & (f < a.pure)).astype(np.float32)).to(dev)
C = torch.stack([pure, mixed, torch.ones(N, device=dev)], -1); res = {'blobs_judged': int(judged.sum()), 'blobs_pure_fruit': int((judged & (f >= a.pure)).sum()), 'blobs_mixed': int((judged & (f >= 0.1) & (f < a.pure)).sum()),
                                                                     'fruit_mass_share_in_pure_blobs': float(fm[judged & (f >= a.pure)].sum() / max(fm.sum(), 1e-9)), 'frames': {}}
acc = [0.0, 0.0, 0.0]
with torch.no_grad():
    for nm in a.frames:
        if nm not in names or not (Path(a.sup) / nm).exists(): continue
        i = names[nm]; cam = ds.cameras[i:i + 1].to(dev); H, Wd = int(cam.height.item()), int(cam.width.item()); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev)
        feat, _, _ = rasterization(means=g['means'].detach(), quats=g['quats'].detach(), scales=torch.exp(g['scales'].detach()), opacities=torch.sigmoid(g['opacities'].detach()).squeeze(-1), colors=C, sh_degree=None,
                                   viewmats=vm, Ks=K, width=Wd, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / nm), np.uint16)).resize((Wd, H), Image.NEAREST)); m = (ids >= FRUIT) & (ids < UNLAB)
        fe = feat[0].cpu().numpy(); tot = fe[..., 2].clip(1e-6); pshare = fe[..., 0] / tot; mshare = fe[..., 1] / tot
        lab_, n = ndimage.label(m); per = []
        for k in range(1, n + 1):
            mk = lab_ == k
            if mk.sum() >= 30: per.append({'px': int(mk.sum()), 'pure_share': float(pshare[mk].mean()), 'mixed_share': float(mshare[mk].mean())})
        res['frames'][nm] = {'fruit_px': int(m.sum()), 'pure_share_over_fruit_px': float(pshare[m].mean()) if m.any() else float('nan'), 'mixed_share_over_fruit_px': float(mshare[m].mean()) if m.any() else float('nan'), 'oranges': per}
        acc[0] += float(pshare[m].sum()); acc[1] += float(mshare[m].sum()); acc[2] += float(m.sum())
res['pure_share_over_all_fruit_px'] = acc[0] / max(acc[2], 1); res['mixed_share_over_all_fruit_px'] = acc[1] / max(acc[2], 1)
print(f"[purity] {Path(a.config).parent.parent.name} step {step}: blobs judged {res['blobs_judged']}, pure-fruit (f>={a.pure}) {res['blobs_pure_fruit']}, mixed {res['blobs_mixed']}; fruit mass in pure blobs {res['fruit_mass_share_in_pure_blobs']:.2f}; "
      f"on the fruit pixels of {len(res['frames'])} frames: rendered mass from pure-fruit blobs {res['pure_share_over_all_fruit_px']:.2f}, from mixed blobs {res['mixed_share_over_all_fruit_px']:.2f} | per orange (>=30 px): "
      + ', '.join(f"{o['pure_share']:.2f}" for nm in res['frames'] for o in res['frames'][nm]['oranges']), flush=True)
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
