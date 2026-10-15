#!/usr/bin/env python3
"""fruit_registry_cl.py — registry v3 (2026-10-15): the same depth-swept 3-D points as fruit_registry_tri.py (its tri_cache.npz), grouped by
agglomerative single linkage with a CANNOT-LINK rule: two detections from the same frame are never the same apple (each mask is one apple,
each apple is at most one mask per frame). v2's merge rule broke this for 65-72 % of its apples (two masks of one frame claimed by one apple).
Pairs within EPS x the median apple radius are merged in order of distance unless the two groups share a frame. An apple = a group seen in
>= MIN_VIEWS frames, counted inside the tree crop (as before). Pre-registered: EPS 2.0, MIN_VIEWS 3 — structural, not fitted to the counts.
system python3:  fruit_registry_cl.py <survey> [--gt 179]"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('survey', type=Path); ap.add_argument('--gt', type=int, default=0); ap.add_argument('--eps', type=float, default=2.0)
ap.add_argument('--min-views', type=int, default=3); ap.add_argument('--crop', type=float, default=0.5); ap.add_argument('--min-support', type=int, default=0); a = ap.parse_args()
S = a.survey.resolve(); Z = np.load(S / 'fruit_registry/tri_cache.npz'); P, detf, rad, sup = Z['Pbest'], Z['detf'], Z['rad'], Z['sup']; N = len(P); R = float(np.median(rad))
use = sup >= a.min_support
pairs = cKDTree(P).query_pairs(a.eps * R, output_type='ndarray'); pairs = pairs[use[pairs[:, 0]] & use[pairs[:, 1]]]
d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1); pairs = pairs[np.argsort(d)]
parent = np.arange(N); fs = [1 << int(f) for f in detf]
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
blocked = 0
for i, j in pairs:
    ri, rj = find(i), find(j)
    if ri == rj: continue
    if fs[ri] & fs[rj]: blocked += 1; continue   # cannot-link: the two groups share a frame
    parent[rj] = ri; fs[ri] |= fs[rj]
roots = np.array([find(i) for i in range(N)]); ur, inv = np.unique(roots, return_inverse=True)
views = np.array([bin(fs[r]).count('1') for r in ur]); cen = np.array([P[inv == g].mean(0) for g in range(len(ur))])
lio = json.loads((S / 'lio_image_poses.json').read_text()); c2w = np.stack([np.asarray(e['transform']) for e in lio]); C = c2w[:, :3, 3]; Fd = c2w[:, :3, 2]; UP = -c2w[:, :3, 1].mean(0); UP /= np.linalg.norm(UP)
A_ = np.zeros((3, 3)); b_ = np.zeros(3)
for c, f in zip(C, Fd / np.linalg.norm(Fd, axis=1, keepdims=True)): Pm = np.eye(3) - np.outer(f, f); A_ += Pm; b_ += Pm @ c
LOOK = np.linalg.solve(A_, b_)
def hd(Y): dd = Y - LOOK; return np.linalg.norm(dd - np.outer(dd @ UP, UP), axis=1)
cam_r = float(np.median(hd(C))); inside = hd(cen) <= a.crop * cam_r; apple = inside & (views >= a.min_views); n = int(apple.sum())
err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
sens = {f'eps{e}_views{v}': None for e in (1.5, 2.0, 3.0) for v in (3, 5, 10)}
print(f"[cl] {S.name}: {N} detections, {len(pairs)} candidate pairs within {a.eps} radii ({blocked} merges blocked by the same-frame rule); groups {len(ur)}, apples (>= {a.min_views} views, in the tree): **{n}**{err}; "
      f"views per apple p10/50/90 {np.percentile(views[apple], 10):.0f}/{np.median(views[apple]):.0f}/{np.percentile(views[apple], 90):.0f}; with >= 10 views {int((apple & (views >= 10)).sum())}, >= 20 views {int((apple & (views >= 20)).sum())}", flush=True)
json.dump({'apples_in_tree': n, 'gt': a.gt or None, 'eps': a.eps, 'min_views': a.min_views, 'views': views[apple].tolist(), 'centroids': cen[apple].tolist(), 'det_group': inv.tolist()},
          open(S / 'fruit_registry/registry_cl.json', 'w'))
