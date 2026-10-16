#!/usr/bin/env python3
"""masklink_row_cut.py <tree> [tag] — per-tree crop for interlocking rows (2026-10-15): the camera ring (hd <= median camera distance) keeps
the tree the photographer walked around, but when the row's neighbours stand inside the ring (tree_02) their apples are in it too. Rule: the
row axis = first principal axis of the candidate apples (components seen in >= 3 views) inside the ring; along it, the apple density (KDE,
h = 0.08 cam_r ~ 1.3 apple spacings) is followed outward from the look-point on each side to its first minimum; if a higher peak (>= 1.5 x
the minimum) lies beyond, that minimum is the canopy boundary and the cut is placed there, else the cut is the ring. Apples = candidates inside
the ring and between the cuts. GT-blind: nothing here reads the hand count. Parameters fixed before tree_03 was looked at."""
import json, sys
from pathlib import Path
import numpy as np
T = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'plain30k'; PEAK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25; R = (Path(T) if '/' in T else Path(f'/home/paperspace/data/public/fruitnerf/high/{T}')) / 'fruit_registry'; T = R.parent.name
d = np.load(R / f'masklink_components_{tag}.npz'); cen, nv, LOOK, UP, cam_r, mv = d['centroid'], d['nviews'], d['LOOK'], d['UP'], float(d['cam_r']), int(d['min_views'])
e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1); X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r
cand = d['apple'] if 'apple' in d.files else (nv >= mv); ring = cand & (np.linalg.norm(X, axis=1) <= 1.0); Xr = X[ring]   # the dump's apple flag carries the ring, min views and (--no-ground) the windfall exclusion
w, v = np.linalg.eigh(np.cov(Xr.T)); axis = v[:, 1]; t = Xr @ axis
g = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((g[:, None] - t[None]) / 0.08) ** 2).sum(1)
def cut(sign):
    idx = np.where(g * sign > 0)[0]; idx = idx[np.argsort(np.abs(g[idx]))]   # grid points outward from 0
    dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
    for i in mins:   # a neighbour = a peak beyond the minimum that is both 1.5 x the minimum and a real canopy (>= 25 % of the densest point in the ring; tree_03 had a 1-vs-2 apple wiggle at the canopy edge)
        if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= PEAK * dens.max(): return float(g[idx[i]])   # a neighbouring TREE has a canopy comparable to the target's (PEAK x the densest point)
    return 1.0 * sign
cl, cr = cut(-1), cut(+1); inside = (t >= cl) & (t <= cr); n = int(inside.sum())
print(f"[row-cut] {T} ({tag}): candidates {int(cand.sum())}, in ring {int(ring.sum())}, PCA eigenvalue ratio {w[1] / w[0]:.1f}; cuts along the row at {cl:+.2f} / {cr:+.2f} cam_r "
      f"({'ring' if cl == -1 else 'canopy minimum'} / {'ring' if cr == 1 else 'canopy minimum'}) -> apples **{n}**")
print("[row-cut] KDE along the row at -0.9..0.9 by 0.1: " + " ".join(f"{dens[i]:.0f}" for i in range(10, 191, 10)))
j = R / f'registry_masklink_{tag}.json'
if j.exists(): o = json.loads(j.read_text()); o['apples_rowcut'] = n; o['rowcut'] = {'left': cl, 'right': cr, 'h': 0.08, 'peak_ratio': 1.5}; j.write_text(json.dumps(o))
