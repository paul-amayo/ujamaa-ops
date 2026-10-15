#!/usr/bin/env python3
"""fruit_registry_count.py — our fruit registry on an image-only survey (FruitNeRF trees, 2026-10-14): every SAM3 'apple' detection (image mode,
one per frame per instance) is lifted to 3-D — mask centroid pixel at the median DA3 depth of the mask's inner pixels (eroded 2 px), pose
c2w_cv from lio_image_poses.json — and the detections are clustered by single linkage at eps = EPS_K x the median apple radius (the apparent
radius sqrt(area/pi) x depth / f, so no metric scale is needed). A cluster with >= MIN_OBS detections is one apple. The target tree is the
vertical axis through the cameras' look-at point (up = mean camera up); apples within CROP x the median horizontal camera distance of that axis
are counted (pre-registered: EPS_K 1.0, MIN_OBS 3, CROP 0.5; the other settings are printed as a sensitivity table, not to pick a winner).
nerf_new pixi env (pycocotools):  fruit_registry_count.py <survey> [--gt 179] [--eps-k 1.0] [--min-obs 3] [--crop 0.5] [--score-min 0.0]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from pycocotools import mask as mask_utils
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('survey', type=Path); ap.add_argument('--gt', type=int, default=0); ap.add_argument('--eps-k', type=float, default=1.0)
ap.add_argument('--min-obs', type=int, default=3); ap.add_argument('--crop', type=float, default=0.5); ap.add_argument('--score-min', type=float, default=0.0); a = ap.parse_args()
S = a.survey.resolve(); OUT = S / 'fruit_registry'; OUT.mkdir(exist_ok=True)
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
tj = json.loads((S / 'transforms.json').read_text()); fx, fy, cx, cy, W, H = tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy'], tj['w'], tj['h']
lio = json.loads((S / 'lio_image_poses.json').read_text()); pose = {int(e['image_idx']): np.asarray(e['transform'], float) for e in lio}
name2idx = {e['image_name']: int(e['image_idx']) for e in lio}
dets = []   # (frame idx, world point, radius scene units, score, area px)
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: (int(f.get('image_idx', f.get('kf_idx'))), f.get('image_name')) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items():
        if int(li) not in l2i: continue
        idx, nm = l2i[int(li)]
        try: idx_d = int(nm.split('_')[1].split('.')[0])
        except Exception: idx_d = idx
        P = pose.get(name2idx.get(nm, idx)); dp = S / 'depth_png' / f'depth_{idx_d}.png'
        if P is None or not dp.exists(): continue
        Z = np.asarray(Image.open(dp), np.float32) / 1000.0
        if Z.shape != (H, W): continue
        for e in ents:
            sc = float(e.get('score', 1.0))
            if sc < a.score_min: continue
            m = mask_utils.decode(e['rle']).astype(bool)
            if m.shape != (H, W) or m.sum() < 20: continue
            inner = ndimage.binary_erosion(m, iterations=2); inner = inner if inner.sum() >= 10 else m
            z = Z[inner]; z = z[z > 0]
            if z.size == 0: continue
            zm = float(np.median(z)); ys, xs = np.where(m); u, v = xs.mean(), ys.mean()
            pc = np.array([(u - cx) * zm / fx, (v - cy) * zm / fy, zm]); pw = P[:3, :3] @ pc + P[:3, 3]
            dets.append((idx, pw, np.sqrt(m.sum() / np.pi) * zm / fx, sc, int(m.sum())))
if not dets: sys.exit('[registry] no detections lifted')
X = np.array([d[1] for d in dets]); R = np.array([d[2] for d in dets]); Rapple = float(np.median(R))
C = np.array([P[:3, 3] for P in pose.values()]); F = np.array([P[:3, 2] for P in pose.values()]); UP = -np.array([P[:3, 1] for P in pose.values()]).mean(0); UP /= np.linalg.norm(UP)
A_ = np.zeros((3, 3)); b_ = np.zeros(3)
for c, f in zip(C, F / np.linalg.norm(F, axis=1, keepdims=True)): Pm = np.eye(3) - np.outer(f, f); A_ += Pm; b_ += Pm @ c
LOOK = np.linalg.solve(A_, b_)
def hdist(Y): d = Y - LOOK; return np.linalg.norm(d - np.outer(d @ UP, UP), axis=1)
cam_r = float(np.median(hdist(C)))
def count(eps_k, min_obs, crop):
    T = cKDTree(X); pairs = T.query_pairs(eps_k * Rapple, output_type='ndarray'); n = len(X)
    import scipy.sparse as sp
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n))
    k, lab = connected_components(G, directed=False); sizes = np.bincount(lab, minlength=k)
    cen = np.array([X[lab == i].mean(0) for i in range(k)]); keep = sizes >= min_obs; inside = hdist(cen) <= crop * cam_r
    return int((keep & inside).sum()), int(keep.sum()), cen, sizes, keep, inside
n_in, n_all, cen, sizes, keep, inside = count(a.eps_k, a.min_obs, a.crop)
res = {'survey': str(S), 'frames_posed': len(pose), 'detections_lifted': len(dets), 'apple_radius_scene_median': Rapple, 'camera_horizontal_distance_median': cam_r,
       'settings': {'eps_k': a.eps_k, 'min_obs': a.min_obs, 'crop': a.crop}, 'count_in_tree': n_in, 'count_all': n_all, 'gt': a.gt or None,
       'sensitivity': {}, 'clusters': [{'centroid': c.tolist(), 'n_obs': int(s), 'h_dist_rel': float(h / cam_r), 'counted': bool(k_ and i_)} for c, s, k_, i_, h in zip(cen, sizes, keep, inside, hdist(cen))]}
for ek in (0.75, 1.0, 1.5):
    for mo in (2, 3, 5):
        for cr in (0.4, 0.5, 0.6):
            res['sensitivity'][f'eps{ek}_obs{mo}_crop{cr}'] = count(ek, mo, cr)[0]
(OUT / 'registry.json').write_text(json.dumps(res, indent=1))
err = f' vs GT {a.gt}: {100 * (n_in - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f"[registry] {S.name}: {len(dets)} apple detections lifted over {len(pose)} posed frames; apple radius {Rapple:.4f} scene units (cameras at {cam_r:.2f} from the tree axis = {cam_r / Rapple:.0f} apple radii); "
      f"clusters with >= {a.min_obs} views: {n_all}, inside the tree (crop {a.crop}): **{n_in}**{err}", flush=True)
s = res['sensitivity']; print('[registry] sensitivity (eps_k x min_obs x crop): ' + ', '.join(f'{k}={v}' for k, v in s.items()), flush=True)
fig, axs = plt.subplots(1, 2, figsize=(13, 6)); e1 = np.cross(UP, [1, 0, 0] if abs(UP[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
def top(Y): d = Y - LOOK; return d @ e1, d @ e2
for ax, (sel, ttl) in zip(axs, ((np.ones(len(cen), bool), 'all clusters'), (keep, f'clusters with >= {a.min_obs} views'))):
    cxs, cys = top(C); ax.scatter(cxs, cys, s=6, c='#888888', label='cameras'); px, py = top(cen[sel])
    ax.scatter(px, py, s=np.clip(sizes[sel], 2, 40), c=np.where(inside[sel], '#e8730c', '#1f77b4'), alpha=0.7, label='apple clusters (orange = counted)')
    ax.add_patch(plt.Circle((0, 0), a.crop * cam_r, fill=False, ls='--', color='k')); ax.set_aspect('equal'); ax.set_title(ttl); ax.legend(loc='upper right', fontsize=8)
fig.suptitle(f"{S.name}: our fruit registry, top-down — {n_in} apples inside the tree crop{err}"); fig.tight_layout(); fig.savefig(OUT / 'registry_topdown.png', dpi=120)
print(f'[registry] -> {OUT}/registry.json, registry_topdown.png', flush=True)
