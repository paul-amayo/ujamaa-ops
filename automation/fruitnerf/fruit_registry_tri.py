#!/usr/bin/env python3
"""fruit_registry_tri.py — the TRIANGULATED fruit registry (2026-10-15; Paul: "do it"): detections are associated across views by reprojection,
not by single-view depth (the depth-lift registry fragmented every apple: 590 / 596 / 1,174 vs 179 / 113 / 291).
  1. depth sweep: for every detection, points along its centroid ray at DA3 depth x [SWEEP_LO, SWEEP_HI] (fine steps, < half an apple radius);
     each point's SUPPORT = number of other frames where its projection lands inside a detection mask AND that frame's DA3 depth there agrees
     within DEPTH_TOL (relative) — so views where the point is hidden behind leaves do not vote. The best-supported depth is the detection's 3-D
     candidate; the detections it lands in are its hits.
  2. greedy registry: candidates in order of support; a candidate whose not-yet-assigned hits span >= MIN_VIEWS frames (own view included)
     founds an apple and claims them; its 3-D centre is the least-squares intersection of the claimed detections' rays.
  3. confirmed apple (the citrus "confirmed fruit" criteria, pre-registered): >= MIN_VIEWS views, median reprojection error <= REPROJ_PX,
     max ray parallax >= PARALLAX_DEG; counted inside the tree crop (0.5 x the median camera distance from the vertical axis through the
     cameras' look-at point). Sensitivity over MIN_VIEWS / REPROJ_PX printed, not used to choose.
nerf_new pixi env (torch, pycocotools):  fruit_registry_tri.py <survey> --gt 179"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mask_utils
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('survey', type=Path); ap.add_argument('--gt', type=int, default=0)
ap.add_argument('--sweep-lo', type=float, default=0.7); ap.add_argument('--sweep-hi', type=float, default=1.3); ap.add_argument('--n-sweep', type=int, default=161)
ap.add_argument('--depth-tol', type=float, default=0.10); ap.add_argument('--min-views', type=int, default=3); ap.add_argument('--reproj-px', type=float, default=6.0)
ap.add_argument('--parallax-deg', type=float, default=2.0); ap.add_argument('--merge-frac', type=float, default=0.3); ap.add_argument('--crop', type=float, default=0.5); a = ap.parse_args()
t0 = time.time(); S = a.survey.resolve(); OUT = S / 'fruit_registry'; OUT.mkdir(exist_ok=True); dev = 'cuda'
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
tj = json.loads((S / 'transforms.json').read_text()); fx, fy, cx, cy, W, H = tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy'], tj['w'], tj['h']
lio = json.loads((S / 'lio_image_poses.json').read_text()); c2w = {int(e['image_idx']): np.asarray(e['transform'], float) for e in lio}
frames = sorted(c2w); fidx = {k: i for i, k in enumerate(frames)}; F = len(frames)
# ---- detections + per-frame instance label maps and depth maps (GPU) ----
L = torch.full((F, H, W), -1, dtype=torch.int32, device=dev); D = torch.zeros((F, H, W), dtype=torch.float16, device=dev)
det_f, det_uv, det_z, det_rpx = [], [], [], []
for k in frames:
    dp = S / 'depth_png' / f'depth_{k}.png'
    if dp.exists(): D[fidx[k]] = torch.from_numpy(np.asarray(Image.open(dp), np.float32) / 1000.0).to(dev).half()
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items():
        k = l2i.get(int(li))
        if k not in fidx: continue
        fi = fidx[k]; Dk = D[fi].float().cpu().numpy(); order = sorted(ents, key=lambda e: e.get('score', 0))   # higher score painted last = wins overlaps
        for e in order:
            m = mask_utils.decode(e['rle']).astype(bool)
            if m.shape != (H, W) or m.sum() < 20: continue
            inner = ndimage.binary_erosion(m, iterations=2); inner = inner if inner.sum() >= 10 else m; z = Dk[inner]; z = z[z > 0]
            if z.size == 0: continue
            i = len(det_f); ys, xs = np.where(m); det_f.append(fi); det_uv.append((xs.mean(), ys.mean())); det_z.append(float(np.median(z))); det_rpx.append(float(np.sqrt(m.sum() / np.pi)))
            L[fi][torch.from_numpy(m).to(dev)] = i
N = len(det_f); det_f = torch.tensor(det_f, device=dev); uv = torch.tensor(det_uv, device=dev, dtype=torch.float32); z0 = torch.tensor(det_z, device=dev); rpx = torch.tensor(det_rpx, device=dev)
Rw = torch.tensor(np.stack([c2w[k][:3, :3] for k in frames]), device=dev, dtype=torch.float32); tw = torch.tensor(np.stack([c2w[k][:3, 3] for k in frames]), device=dev, dtype=torch.float32)
Rc, tc = Rw.transpose(1, 2), -torch.einsum('fij,fj->fi', Rw.transpose(1, 2), tw)   # world -> camera
dirs_c = torch.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, torch.ones(N, device=dev)], -1)   # camera ray at unit z
dirs_w = torch.einsum('nij,nj->ni', Rw[det_f], dirs_c); origins = tw[det_f]
rad_w = rpx * z0 / fx; print(f'[tri] {S.name}: {N} detections over {F} frames, label + depth maps on GPU ({time.time() - t0:.0f}s); apple radius median {float(rad_w.median()):.4f}', flush=True)
def hits_for(P):   # P [M, 3] world -> [M, F] hit detection id (-1 none), depth-consistent
    M = P.shape[0]; out = torch.full((M, F), -1, dtype=torch.int32, device=dev)
    for j in range(F):
        pc = P @ Rc[j].T + tc[j]; zj = pc[:, 2]; u = (fx * pc[:, 0] / zj.clamp(min=1e-6) + cx).round().long(); v = (fy * pc[:, 1] / zj.clamp(min=1e-6) + cy).round().long()
        ok = (zj > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H); uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        lab = L[j][vv, uu]; dz = D[j][vv, uu].float(); ok = ok & (lab >= 0) & ((zj - dz).abs() <= a.depth_tol * zj)
        out[:, j] = torch.where(ok, lab, torch.full_like(lab, -1))
    return out
# ---- 1. depth sweep: best-supported depth per detection ----
scales = torch.linspace(a.sweep_lo, a.sweep_hi, a.n_sweep, device=dev); best_s = torch.ones(N, device=dev); best_sup = torch.zeros(N, dtype=torch.int32, device=dev)
for c0 in range(0, N, 1500):
    idx = torch.arange(c0, min(N, c0 + 1500), device=dev); M = idx.numel()
    P = origins[idx, None, :] + (z0[idx, None, None] * scales[None, :, None]) * dirs_w[idx, None, :]   # [M, S, 3]
    sup = torch.zeros(M, a.n_sweep, dtype=torch.int32, device=dev); Pf = P.reshape(-1, 3)
    for j in range(F):
        pc = Pf @ Rc[j].T + tc[j]; zj = pc[:, 2]; u = (fx * pc[:, 0] / zj.clamp(min=1e-6) + cx).round().long(); v = (fy * pc[:, 1] / zj.clamp(min=1e-6) + cy).round().long()
        ok = (zj > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H); uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        lab = L[j][vv, uu]; dz = D[j][vv, uu].float(); ok = ok & (lab >= 0) & ((zj - dz).abs() <= a.depth_tol * zj)
        sup += ok.view(M, a.n_sweep).int()
    tie = -(scales - 1).abs()[None, :] * 1e-3   # ties -> nearest to the DA3 depth
    bi = (sup.float() + tie).argmax(1); best_s[idx] = scales[bi]; best_sup[idx] = sup[torch.arange(M, device=dev), bi]
Pbest = origins + (z0 * best_s)[:, None] * dirs_w
print(f'[tri] depth sweep done ({time.time() - t0:.0f}s): support (frames incl. own) median {float(best_sup.float().median()):.0f}, p90 {float(best_sup.float().quantile(0.9)):.0f}; best depth / DA3 depth median {float(best_s.median()):.3f} (p10 {float(best_s.quantile(0.1)):.3f}, p90 {float(best_s.quantile(0.9)):.3f})', flush=True)
HIT = torch.cat([hits_for(Pbest[c:c + 4000]) for c in range(0, N, 4000)]).cpu().numpy(); detf = det_f.cpu().numpy()
np.savez_compressed(OUT / 'tri_cache.npz', Pbest=Pbest.cpu().numpy(), HIT=HIT, detf=detf, sup=best_sup.cpu().numpy(), rad=rad_w.cpu().numpy(), origins=origins.cpu().numpy(),
                    dirs=dirs_w.cpu().numpy(), uv=uv.cpu().numpy(), Rc=Rc.cpu().numpy(), tc=tc.cpu().numpy(), K=np.array([fx, fy, cx, cy]), frames=np.array(frames))   # for re-grouping without the 6-min load
# ---- 2. greedy registry ----
assigned = np.full(N, -1); apples = []
for i in np.argsort(-best_sup.cpu().numpy(), kind='stable'):
    if assigned[i] >= 0: continue
    h = HIT[i]; h = np.unique(np.concatenate([h[h >= 0], [i]])); own = assigned[h]
    claimed = own[own >= 0]
    if claimed.size:   # duplicate of an apple that already claims some of these detections -> merge into the apple holding the most
        A = np.bincount(claimed).argmax()
        if (claimed == A).sum() >= a.merge_frac * h.size:
            free = h[own < 0]; assigned[free] = A; apples[A] = np.concatenate([apples[A], free]); continue
    h = h[own < 0]
    if len(np.unique(detf[h])) < a.min_views: continue
    aid = len(apples); assigned[h] = aid; apples.append(h)
# ---- 3. triangulate + confirm ----
O = origins.cpu().numpy(); Dw = dirs_w.cpu().numpy(); Dw = Dw / np.linalg.norm(Dw, axis=1, keepdims=True); UVn = uv.cpu().numpy(); Rcn, tcn = Rc.cpu().numpy(), tc.cpu().numpy()
rec = []
for aid, h in enumerate(apples):
    A_ = np.zeros((3, 3)); b_ = np.zeros(3)
    for i in h: Pm = np.eye(3) - np.outer(Dw[i], Dw[i]); A_ += Pm; b_ += Pm @ O[i]
    try: X = np.linalg.solve(A_, b_)
    except np.linalg.LinAlgError: continue
    errs = []
    for i in h:
        j = detf[i]; pc = Rcn[j] @ X + tcn[j]
        errs.append(np.hypot(fx * pc[0] / pc[2] + cx - UVn[i, 0], fy * pc[1] / pc[2] + cy - UVn[i, 1]) if pc[2] > 0 else 1e9)
    cosm = np.clip(Dw[h] @ Dw[h].T, -1, 1); par = float(np.degrees(np.arccos(cosm.min())))
    rec.append({'xyz': X.tolist(), 'views': int(len(np.unique(detf[h]))), 'reproj_med_px': float(np.median(errs)), 'parallax_deg': par, 'dets': [int(i) for i in h]})
C = np.stack([c2w[k][:3, 3] for k in frames]); Fd = np.stack([c2w[k][:3, 2] for k in frames]); UP = -np.stack([c2w[k][:3, 1] for k in frames]).mean(0); UP /= np.linalg.norm(UP)
A_ = np.zeros((3, 3)); b_ = np.zeros(3)
for c, f in zip(C, Fd / np.linalg.norm(Fd, axis=1, keepdims=True)): Pm = np.eye(3) - np.outer(f, f); A_ += Pm; b_ += Pm @ c
LOOK = np.linalg.solve(A_, b_)
def hd(Y): d = Y - LOOK; return np.linalg.norm(d - np.outer(d @ UP, UP), axis=1)
cam_r = float(np.median(hd(C))); X = np.array([r['xyz'] for r in rec]) if rec else np.zeros((0, 3)); inside = hd(X) <= a.crop * cam_r if len(X) else np.zeros(0, bool)
def n_conf(mv, rp, pa): return int(sum(1 for r, ins in zip(rec, inside) if ins and r['views'] >= mv and r['reproj_med_px'] <= rp and r['parallax_deg'] >= pa))
n = n_conf(a.min_views, a.reproj_px, a.parallax_deg); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
sens = {f'views{mv}_reproj{rp}': n_conf(mv, rp, a.parallax_deg) for mv in (3, 5, 8) for rp in (4.0, 6.0, 10.0)}
for r, ins in zip(rec, inside): r['inside'] = bool(ins); r['confirmed'] = bool(ins and r['views'] >= a.min_views and r['reproj_med_px'] <= a.reproj_px and r['parallax_deg'] >= a.parallax_deg)
json.dump({'survey': str(S), 'detections': N, 'frames': F, 'apples_founded': len(apples), 'confirmed_in_tree': n, 'gt': a.gt or None, 'settings': vars(a) | {'survey': str(S)}, 'sensitivity': sens,
           'assigned_frac': float((assigned >= 0).mean()), 'apples': rec}, open(OUT / 'registry_tri.json', 'w'), indent=1, default=str)
cr = [r for r in rec if r['confirmed']]
print(f"[tri] {S.name}: {len(apples)} apples founded ({100 * (assigned >= 0).mean():.0f} % of detections claimed); confirmed in the tree (>= {a.min_views} views, reproj <= {a.reproj_px} px, parallax >= {a.parallax_deg} deg): **{n}**{err}; "
      f"confirmed apples: views median {np.median([r['views'] for r in cr]) if cr else 0:.0f}, reprojection median {np.median([r['reproj_med_px'] for r in cr]) if cr else 0:.1f} px ({time.time() - t0:.0f}s)", flush=True)
print('[tri] sensitivity (min views x reprojection px): ' + ', '.join(f'{k}={v}' for k, v in sens.items()), flush=True)
fig, ax = plt.subplots(1, 1, figsize=(7, 7)); e1 = np.cross(UP, [1, 0, 0] if abs(UP[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
tp = lambda Y: ((Y - LOOK) @ e1, (Y - LOOK) @ e2); ax.scatter(*tp(C), s=6, c='#888888', label='cameras')
if len(X):
    cf = np.array([r['confirmed'] for r in rec]); ax.scatter(*tp(X[~cf]), s=6, c='#1f77b4', alpha=0.5, label='not confirmed / outside'); ax.scatter(*tp(X[cf]), s=10, c='#e8730c', label=f'confirmed apples in the tree ({n})')
ax.add_patch(plt.Circle((0, 0), a.crop * cam_r, fill=False, ls='--', color='k')); ax.set_aspect('equal'); ax.legend(fontsize=8); ax.set_title(f'{S.name}: triangulated registry — {n} apples{err}', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'registry_tri_topdown.png', dpi=120); print(f'[tri] -> {OUT}/registry_tri.json, registry_tri_topdown.png', flush=True)
