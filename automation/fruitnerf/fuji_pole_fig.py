#!/usr/bin/env python3
"""fuji_pole_fig.py --along 4.54 --out <png> (2026-10-16) — qualitative view of one trunk column: the photos that see it best on each side with the
SAM3 'tree trunk' masks (red) and the column's trunk-labelled gaussians projected (cyan = 0.3-1.5 m band, yellow = above 4.5 m = pole), plus a
side view (along the row vs height) of the trunk gaussians around the column. nerf_new pixi env (pycocotools)."""
import argparse, json, glob, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
sys.path.insert(0, str(Path(__file__).resolve().parent)); from fuji_region_check import read_cameras, read_images
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--along', type=float, required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', default='plain30k'); ap.add_argument('--align', default='align_flip'); ap.add_argument('--per-side', type=int, default=2); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R = ROOT / 'high'
cloud = np.load(ROOT / 'their_cloud_500k.npy').astype(np.float64); c0 = cloud.mean(0); w_, v_ = np.linalg.eigh(np.cov((cloud[:, :2] - c0[:2]).T)); row2 = v_[:, -1]; acr2 = np.array([-row2[1], row2[0]])
def rowframe(P): return np.stack([(P[:, :2] - c0[:2]) @ row2, (P[:, :2] - c0[:2]) @ acr2, P[:, 2]], 1)
panels = []; side_pts = []
for side in ('east', 'west'):
    S = R / side; z = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z['s']), z['R'], z['t']; dpt = json.loads((S / 'outputs/sf_30000_bilateral/splatfacto/run/dataparser_transforms.json').read_text()); T = np.array(dpt['transform']); sc = float(dpt['scale'])
    tc = np.load(S / f'fruit_registry/trunk_census_{a.tag}.npz'); Pg = (s * (Rm @ (((tc['P'].astype(np.float64) / sc) - T[:, 3]) @ T[:, :3]).T)).T + t; rf = rowframe(Pg)
    Mg = (s * (Rm @ (((np.load(S / 'fruit_registry/means_plain30k.npy').astype(np.float64) / sc) - T[:, 3]) @ T[:, :3]).T)).T + t; Mrf = rowframe(Mg); ground = np.percentile(Mrf[np.abs(Mrf[:, 1]) < 1.5, 2], 2); h = rf[:, 2] - ground
    col = (np.abs(rf[:, 0] - a.along) < 0.25) & (np.abs(rf[:, 1]) < 0.6); low = col & (h >= 0.3) & (h <= 1.5); high = col & (h > 4.5); wide = (np.abs(rf[:, 0] - a.along) < 0.8) & (np.abs(rf[:, 1]) < 0.6); side_pts.append((side, rf[wide, 0], h[wide]))
    print(f"[pole] {side}: column at {a.along:+.2f}: {int(col.sum())} trunk gaussians, {int(low.sum())} in 0.3-1.5 m, {int(high.sum())} above 4.5 m (top {h[col].max():.2f} m above our ground)")
    ims = read_images(S / 'sparse/0/images.bin'); cams = read_cameras(S / 'sparse/0/cameras.bin'); masks = {}
    for cm in sorted((S / 'prod/bateleur/sam3_trunk').glob('clip_*/frame_entries.json')):
        meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
        for li, ents in meta['frame_entries'].items(): masks[l2i[int(li)]] = [mask_utils.decode(e['rle']).astype(bool) for e in ents]
    scored = []
    for nm, (Rq, tt, cid) in ims.items():
        k = int(nm.split('_')[1].split('.')[0]); model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]; C = s * (Rm @ (-Rq.T @ tt)) + t; Rot = Rq @ Rm.T; crf = rowframe(C[None])[0]
        if abs(crf[0] - a.along) > 1.5: continue
        def proj(P):
            pc = (1.0 / s) * (Rot @ (P - C).T).T; zc = pc[:, 2]; ok = zc > 0.3; xn, yn = pc[ok, 0] / zc[ok], pc[ok, 1] / zc[ok]; r2 = xn ** 2 + yn ** 2; rad = 1 + k1 * r2 + k2 * r2 ** 2
            u = fx * (xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)) + cx; v = fy * (yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn) + cy; return u, v, (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, ins = proj(Pg[low]); frac = ins.mean() if len(ins) else 0; scored.append((frac, -abs(crf[0] - a.along), k, nm, proj))
    scored.sort(reverse=True)
    for frac, _, k, nm, proj in scored[:a.per_side]:
        photo = np.asarray(Image.open(S / 'images' / nm).convert('RGB')); ov = photo.astype(float)
        for mk in masks.get(k, []):
            if mk.shape == photo.shape[:2]: ov[mk] = 0.6 * ov[mk] + 0.4 * np.array([255, 0, 0])
        ul, vl, il = proj(Pg[low]); uh, vh, ih = proj(Pg[high]); panels.append((f"{side} image_{k}: trunk masks (red), column gaussians 0.3-1.5 m (cyan) / above 4.5 m (yellow); {100 * frac:.0f} % of the band in frame", ov.astype(np.uint8), (ul[il], vl[il]), (uh[ih], vh[ih])))
n = len(panels); fig = plt.figure(figsize=(20, 5 * ((n + 1) // 2) + 5)); gs = fig.add_gridspec((n + 1) // 2 + 1, 2)
for i, (title, ov, lo, hi) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 2, i % 2]); ax.imshow(ov); ax.scatter(lo[0], lo[1], s=2, c='cyan'); ax.scatter(hi[0], hi[1], s=2, c='yellow'); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[-1, :])
for side, x, hh in side_pts: ax.scatter(x, hh, s=1, label=f'{side} trunk gaussians', alpha=0.5)
ax.axhline(4.5, color='k', ls='--', lw=0.8); ax.axvline(a.along, color='k', lw=0.8); ax.set_xlabel('along the row [m]'); ax.set_ylabel('height above our ground [m]'); ax.set_title(f'side view of trunk-labelled gaussians around the column at {a.along:+.2f} m (dashed = 4.5 m pole rule)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(a.out, dpi=70); print('[pole] wrote', a.out)
