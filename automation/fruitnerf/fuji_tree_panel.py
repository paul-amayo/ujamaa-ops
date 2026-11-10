#!/usr/bin/env python3
"""fuji_tree_panel.py <out.npz> [--counts tree_counts_flip_d36.json] [--diag <png>] (2026-11-10) — the Fuji-SfM panel of the HiGH cross-site figure
(sites_strip.py --fuji): the fruit level of the hierarchy, i.e. the apples our registry assigns to ONE tree, on one of the dataset's photographs,
coloured against the ground-truth apples (true positive / false positive / miss).
  tree   picked from the counts json alone, before any image: the tree whose per-tree F1 is nearest the median per-tree F1; its P / R / F1 are
         printed from the json, and the TP / FP / FN sets are recomputed by fuji_tree_counts.py's rule (merged apples -> nearest trunk along
         the row; GT tree <-> trunk one-to-one by nearest median within 0.6 m; TP = an apple of the tree's trunk matched at 10 cm (pair) to a GT
         apple of that tree) and asserted equal to the json's counts.
  camera the dataset ships its georeferenced cloud and GT but no camera poses, so the photo's camera is our COLMAP registration of the dataset's
         own photographs (high/<side>/sparse/0) carried into the dataset's frame by align_flip (similarity to their cloud), raw-image OPENCV
         distortion; occlusion test against their full cloud: an apple is hidden when the ray meets their cloud before 85 % of its depth.
  photo  align_flip is good to ~one apple in some photos and ~two to three apple radii off in others (east image_186: 9 % of visible GT apples
         within one mask radius, west image_100: 63 %), so a photo must first pass fuji_gt_on_masks.py's registration check, blind to the tree's
         classes: >= --reg of all visible GT apples within one radius of one of our SAM3 apple masks. Among those, the photograph (east or west)
         in which the most of the tree's scored apples (TP | FP | FN) are visible; apples hidden from that view
         are not drawn (counts printed). TP and FP are drawn at our apple, FN at the GT apple; disc radius = 4 cm at the apple's depth.
system python3 (numpy, scipy, PIL)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); from fuji_region_check import read_cameras, read_images
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--counts', default='/home/paperspace/data/public/fuji_sfm/tree_counts_flip_d36.json')
ap.add_argument('--merged', default='/home/paperspace/data/public/fuji_sfm/merged_flip_r10.npz'); ap.add_argument('--diag', default=None, help='full photo with every drawn apple ringed, to check the projection')
ap.add_argument('--top', type=int, default=60, help='photos tried: the ones with the most of the tree\'s scored apples in frame'); ap.add_argument('--reg', type=float, default=0.5, help='registration gate: share of visible GT apples within one mask radius of a mask centre'); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R_APPLE = 0.04
J = json.load(open(a.counts)); rows = J['rows']; f1s = np.array([r['f1'] for r in rows]); med = float(np.median(f1s)); row = rows[int(np.argmin(np.abs(f1s - med)))]; TREE = row['tree']
print(f"[fuji-panel] per-tree F1 median {med:.3f} over {len(rows)} trees; picked {TREE}: F1 {row['f1']:.3f} (P {row['p']:.3f}, R {row['r']:.3f}; ours {row['ours']}, GT {row['gt']}, TP {row['tp']})")
# ---- recompute the tree's TP / FP / FN by fuji_tree_counts.py's rule ----
c500 = np.load(ROOT / 'their_cloud_500k.npy').astype(np.float64); c0 = c500.mean(0); w_, v_ = np.linalg.eigh(np.cov((c500[:, :2] - c0[:2]).T)); row2 = v_[:, -1]
M = np.load(a.merged); P, pair, GT, GT_tree = M['P'], M['pair'], M['GT'], M['GT_tree']; along = lambda X: (X[:, :2] - c0[:2]) @ row2
TR = np.array(J['trunks_along']); assign = np.argmin(np.abs(along(P)[:, None] - TR[None, :]), axis=1)
trees = sorted(set(GT_tree)); gmed = np.array([np.median(along(GT[GT_tree == t_])) for t_ in trees]); d = np.abs(gmed[:, None] - TR[None, :]); used = set(); t2k = {}
for k in np.argsort(d, axis=None):
    i, j = divmod(int(k), len(TR))
    if d[i, j] > 0.6: break
    if trees[i] in t2k or j in used: continue
    t2k[trees[i]] = j; used.add(j)
gi = np.where(GT_tree == TREE)[0]; mine = assign == t2k[TREE]; tp = mine & np.isin(pair, gi) & (pair >= 0); fp = mine & ~tp; fn = np.setdiff1d(gi, pair[tp])
assert (int(mine.sum()), int(tp.sum()), len(gi)) == (row['ours'], row['tp'], row['gt']), f"recomputed ours/TP/GT {int(mine.sum())}/{int(tp.sum())}/{len(gi)} != json {row['ours']}/{row['tp']}/{row['gt']}"
X = np.concatenate([P[tp], P[fp], GT[fn]]); CLS = np.concatenate([np.zeros(tp.sum(), int), np.ones(fp.sum(), int), np.full(len(fn), 2)])
print(f"[fuji-panel] recomputed {TREE}: TP {tp.sum()}, FP {fp.sum()}, FN {len(fn)} (matches the json)")
# ---- every photograph's view of those apples ----
print('[fuji-panel] loading their cloud for the occlusion test ...', flush=True); cloud = cKDTree(np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64))
def project(side, name, X=X):
    S = ROOT / 'high' / side; z_ = np.load(S / 'fruit_registry/align_flip.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; Rq, tt, cid = IMS[side][name]; model, W, H, par = CAMS[side][cid]
    fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]; C = s * (Rm @ (-Rq.T @ tt)) + t; pc = (1.0 / s) * ((Rq @ Rm.T) @ (X - C).T).T; z = pc[:, 2]; ok = z > 0.5
    xn, yn = pc[:, 0] / np.where(ok, z, 1), pc[:, 1] / np.where(ok, z, 1); r2 = xn ** 2 + yn ** 2; rad = 1 + k1 * r2 + k2 * r2 ** 2
    u = fx * (xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)) + cx; v = fy * (yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn) + cy
    inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H); dist = np.linalg.norm(X - C, axis=1); return C, u, v, inside, dist, fx * R_APPLE / dist, (W, H)
def visible(C, idx, dist, X=X, need_hit=False):
    rays = (X[idx] - C) / dist[idx, None]; steps = np.arange(0.3, 10.0, 0.02); Q = C[None, None, :] + rays[:, None, :] * steps[None, :, None]
    dd, _ = cloud.query(Q.reshape(-1, 3), k=1, distance_upper_bound=0.03, workers=8); h = np.isfinite(dd).reshape(len(idx), len(steps)); first = np.where(h.any(1), steps[h.argmax(1)], np.nan)
    if need_hit: return np.isfinite(first) & (first >= 0.85 * dist[idx])   # fuji_gt_on_masks.py's rule (the GT apples are on their cloud)
    return ~(first < 0.85 * dist[idx])   # not occluded: no cloud surface before 85 % of the apple's depth (an apple off every surface still counts)
def registration(side, name):
    """fuji_gt_on_masks.py's check, blind to the tree's TP / FP / FN: every GT apple visible in the photo vs the centres of our SAM3 apple masks of that photo;
    the share within one mask radius of a centre and the median distance (px) over the median mask radius"""
    C, u, v, inside, dist, _, _ = project(side, name, GT); idx = np.where(inside)[0]; vis = visible(C, idx, dist, GT, need_hit=True)
    mk = np.asarray(Image.open(ROOT / 'high' / side / 'fruit_masks' / name)); vals, cnts = np.unique(mk, return_counts=True); fruit = mk == vals[np.argmin(cnts)]; lab, n = ndimage.label(fruit)
    cents = np.array(ndimage.center_of_mass(fruit, lab, range(1, n + 1))); areas = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))); k_ = areas >= 30; cents, radii = cents[k_], np.sqrt(areas[k_] / np.pi)
    if vis.sum() < 10 or len(cents) < 10: return 0.0, np.inf, int(vis.sum())
    dm, jm = cKDTree(cents[:, ::-1]).query(np.stack([u[idx][vis], v[idx][vis]], 1), k=1); return float((dm <= radii[jm]).mean()), float(np.median(dm) / np.median(radii)), int(vis.sum())
IMS = {s_: read_images(ROOT / 'high' / s_ / 'sparse/0/images.bin') for s_ in ('east', 'west')}; CAMS = {s_: read_cameras(ROOT / 'high' / s_ / 'sparse/0/cameras.bin') for s_ in ('east', 'west')}
cand = sorted(((int(project(s_, n_)[3].sum()), s_, n_) for s_ in IMS for n_ in IMS[s_]), reverse=True)[:a.top]; best = None
for n_in, side, name in cand:
    C, u, v, inside, dist, rpx, WH = project(side, name); idx = np.where(inside)[0]; vis = visible(C, idx, dist); nv = int(vis.sum()); f1r, dr_, ngt = registration(side, name); ok = f1r >= a.reg
    print(f"[fuji-panel]   {side}/{name}: tree apples in frame {n_in}, visible {nv}; registration: {100 * f1r:.0f} % of {ngt} visible GT apples within one mask radius, median offset {dr_:.2f} radii{'' if ok else '  (fails)'}", flush=True)
    if ok and (best is None or nv > best[0]): best = (nv, side, name, idx[vis], n_in, f1r, dr_)
if best is None: raise SystemExit(f'[fuji-panel] no photo among the top {a.top} passes the registration check (>= {100 * a.reg:.0f} % of visible GT apples within one mask radius); stop')
nv, side, name, keep, n_in, f1r, dr_ = best; C, u, v, inside, dist, rpx, WH = project(side, name)
src = (ROOT / 'high' / side / 'source_names.txt').read_text().split()[int(name.split('_')[1].split('.')[0])]
cnt = lambda m: [int((CLS[m] == c).sum()) for c in range(3)]
print(f"[fuji-panel] photo {side}/{name} (dataset {src}; registration {100 * f1r:.0f} % within one radius, median offset {dr_:.2f} radii): {n_in} of the tree's {len(X)} scored apples in frame, {nv} visible; drawn TP/FP/FN {cnt(keep)} of {cnt(np.arange(len(X)))}")
photo = np.asarray(Image.open(ROOT / 'high' / side / 'images' / name).convert('RGB'))
np.savez_compressed(a.out, photo=photo, u=u[keep], v=v[keep], cls=CLS[keep], rpx=rpx[keep], tree=TREE, f1=row['f1'], p=row['p'], r=row['r'], tp=row['tp'], ours=row['ours'], gt=row['gt'],
                    f1_median=med, reg_within=f1r, reg_offset=dr_, side=side, image=name, source=src, drawn=np.array(cnt(keep)), total=np.array(cnt(np.arange(len(X)))), counts=a.counts)
print('[fuji-panel] wrote', a.out)
if a.diag:
    from PIL import ImageDraw
    im = Image.fromarray(photo); dr = ImageDraw.Draw(im); col = [(0, 255, 0), (255, 140, 0), (255, 0, 0)]
    for uu, vv, c, r_ in zip(u[keep], v[keep], CLS[keep], rpx[keep]): dr.ellipse([uu - r_, vv - r_, uu + r_, vv + r_], outline=col[c], width=2)
    im.save(a.diag); print('[fuji-panel] diag', a.diag)
