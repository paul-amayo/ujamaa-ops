#!/usr/bin/env python3
"""fruit_registry_maskgraph_count.py <tree> [tag] [--theta T] — stage 2 of the mask-graph registry: masks are nodes; two masks of DIFFERENT
views are the same apple when the gaussians that render them overlap (cosine similarity of their mass vectors >= theta); two masks of the
SAME view are never the same apple (exact cannot-link). Greedy agglomeration by descending similarity under that constraint; an apple =
a cluster of masks from >= 3 views, placed at the mass-weighted mean of its masks' lifted centroids (masklink_votes_<tag>.npz), counted
inside the camera ring and the row cut (same rule as masklink_row_cut.py). Prints the count for a sweep of theta (GT-blind curve) and
writes registry_maskgraph_<tag>.json for --theta. Pieced-apple indicator: apple pairs closer than one apple radius (0.0125 cam_r)."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--theta', type=float, default=0.2); ap.add_argument('--gt', type=int, default=0); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); d = np.load(R / f'maskgraph_{a.tag}.npz'); v = np.load(R / f'masklink_votes_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz')
mrow, gid, w, mview, mmass = d['mrow'], d['gid'], d['w'], d['mview'], d['mmass']; nM = len(mview); N = int(d['N'])
C = csr_matrix((w, (mrow, gid)), shape=(nM, N)); nrm = np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel()); nrm[nrm == 0] = 1; Cn = csr_matrix(C.multiply(1 / nrm[:, None]))
O = (Cn @ Cn.T).tocoo(); keep = (O.row < O.col) & (mview[O.row] != mview[O.col]) & (O.data > 0.02); ea, eb, es = O.row[keep], O.col[keep], O.data[keep]; order = np.argsort(-es); ea, eb, es = ea[order], eb[order], es[order]
# lifted centroids per (view, mask) from the votes cache
mk = v['masks']; key = (mk[:, 0].astype(np.int64) * 100000 + mk[:, 1].astype(np.int64)); pos = dict(zip(key.tolist(), mk[:, 2:5])); P = np.array([pos.get(int(vv) * 100000 + int(ll), [np.nan] * 3) for vv, ll in zip(mview, d['mlab'])], np.float64)
LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
def cluster(theta):
    parent = np.arange(nM); views = [{int(x)} for x in mview]
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for A_, B_, s_ in zip(ea, eb, es):
        if s_ < theta: break
        ra, rb = find(A_), find(B_)
        if ra == rb or (views[ra] & views[rb]): continue
        if len(views[ra]) < len(views[rb]): ra, rb = rb, ra
        parent[rb] = ra; views[ra] |= views[rb]; views[rb] = set()
    roots = np.array([find(x) for x in range(nM)]); return roots
def count(theta, dump=False):
    roots = cluster(theta); uniq, inv = np.unique(roots, return_inverse=True); k = len(uniq); nv_ = np.zeros(k, int)
    for r_ in range(k): pass
    nviews = np.array([len(set(mview[inv == r_].tolist())) for r_ in range(k)]); mass = np.bincount(inv, weights=mmass, minlength=k)
    ok = ~np.isnan(P[:, 0]); cen = np.zeros((k, 3))
    for q in range(3): cen[:, q] = np.bincount(inv[ok], weights=(P[ok, q] * mmass[ok]), minlength=k) / np.maximum(np.bincount(inv[ok], weights=mmass[ok], minlength=k), 1e-9)
    X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r; cand = nviews >= 3; ring = cand & (np.linalg.norm(X, axis=1) <= 1.0)
    # row cut as in masklink_row_cut.py
    Xr = X[ring]; wv, vv = np.linalg.eigh(np.cov(Xr.T)); axis = vv[:, 1]; t = Xr @ axis; gr = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((gr[:, None] - t[None]) / 0.08) ** 2).sum(1)
    def cut(sign):
        idx = np.where(gr * sign > 0)[0]; idx = idx[np.argsort(np.abs(gr[idx]))]; dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
        for i in mins:
            if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= 0.25 * dens.max(): return float(gr[idx[i]])
        return 1.0 * sign
    cl, cr = cut(-1), cut(1); inside = (t >= cl) & (t <= cr); n_ring = int(ring.sum()); n_row = int(inside.sum())
    if dump:
        from scipy.spatial import cKDTree
        A_ = cen[ring][inside]; dd, _ = cKDTree(A_).query(A_, k=2); close = int((dd[:, 1] / cam_r < 0.0125).sum())
        return n_ring, n_row, k, nviews, mass, cen, ring, inv, (cl, cr), close
    return n_ring, n_row
print(f'[maskgraph] {a.tree} ({a.tag}): {nM} masks, {len(es)} cross-view edges (cos > 0.02); count vs theta (ring / row cut):', flush=True)
print('[maskgraph]   ' + '  '.join(f'θ{th:.2f}: {count(th)[0]}/{count(th)[1]}' for th in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)), flush=True)
n_ring, n_row, k, nviews, mass, cen, ring, inv, cuts, close = count(a.theta, dump=True); err = f' vs GT {a.gt}: {100 * (n_row - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f"[maskgraph] {a.tree} ({a.tag}) θ={a.theta}: {k} clusters, {int((nviews >= 3).sum())} with >= 3 views, {n_ring} in the ring, row cut {cuts[0]:+.2f}/{cuts[1]:+.2f} -> apples **{n_row}**{err}; views per apple median {np.median(nviews[ring]):.0f}; apple pairs closer than one apple radius {close}", flush=True)
json.dump({'tag': a.tag, 'theta': a.theta, 'apples': n_row, 'apples_ring': n_ring, 'gt': a.gt or None, 'clusters': int(k)}, open(R / f'registry_maskgraph_{a.tag}.json', 'w'))
np.savez_compressed(R / f'maskgraph_clusters_{a.tag}.npz', inv=inv, nviews=nviews, cen=cen, ring=ring, theta=a.theta)
