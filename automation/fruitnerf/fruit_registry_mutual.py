#!/usr/bin/env python3
"""fruit_registry_mutual.py <tree> [tag] [--theta T] [--window W] [--gt N] — registry by MUTUAL BEST MATCHES between nearby views (2026-10-15 18:0x).
Measured on tree_01: one apple's per-view 3-D estimates scatter by 1.4-1.5 apple radii (both the rendered-depth lift and the contributor
centroid), and its mask centroids are 0.8 radii from any single 3-D point — the geometry is noisy at the apple scale, so 3-D gating cannot
separate touching apples. The splat's RENDERING is consistent even where its depth is not, so masks are matched the way the splat renders
them: two masks in views A and B share rendering gaussians (cosine of their mass vectors from maskgraph_<tag>.npz). Links are formed only
between views within --window steps in orbit order and only when the two masks are each other's best match (mutual, exclusive per view pair)
with cosine >= theta. Apples = connected components with >= 3 views; position = median lifted centroid; camera ring + row cut as before.
Diagnostics (GT-free): masks per view per component (1 = no merge), views per apple, NN distances in radii, masks absorbed. venv leiden."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--theta', type=float, default=0.3); ap.add_argument('--window', type=int, default=3); ap.add_argument('--gt', type=int, default=0); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); d = np.load(R / f'maskgraph_{a.tag}.npz'); v = np.load(R / f'masklink_votes_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz'); cams = np.load(R / f'cameras_{a.tag}.npz')
mrow, gid, w, mview, mlab, mmass = d['mrow'], d['gid'], d['w'], d['mview'], d['mlab'], d['mmass']; nM = len(mview); N = int(d['N']); V = len(cams['names'])
LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); RAD = 0.0125 * cam_r; e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
VM = cams['viewmat']; Cc = np.array([-VM[i, :3, :3].T @ VM[i, :3, 3] for i in range(V)]); az = np.arctan2((Cc - LOOK) @ e2, (Cc - LOOK) @ e1); order = np.argsort(az); rank = np.empty(V, int); rank[order] = np.arange(V)
C = csr_matrix((w, (mrow, gid)), shape=(nM, N)); nrm = np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel()); nrm[nrm == 0] = 1; Cn = csr_matrix(C.multiply(1 / nrm[:, None])); O = (Cn @ Cn.T).tocsr()
mk = v['masks']; pos = {(int(r[0]), int(r[1])): r[2:5] for r in mk}; P = np.array([pos.get((int(vv), int(ll)), [np.nan] * 3) for vv, ll in zip(mview, mlab)])
by_view = [np.where(mview == i)[0] for i in range(V)]
def links(theta, window):
    ea, eb = [], []
    for i in range(V):
        A = by_view[i]
        if not len(A): continue
        for s in range(1, window + 1):
            j = order[(rank[i] + s) % V]; B = by_view[j]
            if not len(B): continue
            sub = O[A][:, B].toarray()   # |A| x |B| cosines
            if sub.size == 0: continue
            ba = sub.argmax(1); bb = sub.argmax(0)
            for ia, ib in enumerate(ba):
                if bb[ib] == ia and sub[ia, ib] >= theta: ea.append(A[ia]); eb.append(B[ib])
    return np.array(ea), np.array(eb)
def count(theta, window, dump=False):
    ea, eb = links(theta, window); G = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(nM, nM)); k, comp = connected_components(G, directed=False)
    nviews = np.zeros(k, int); mpv = np.zeros(k)
    for b in range(k): pass
    vs_counts = {}
    for m_, cc in enumerate(comp): vs_counts.setdefault(cc, {}).setdefault(int(mview[m_]), 0); vs_counts[cc][int(mview[m_])] += 1
    for cc, dct in vs_counts.items(): nviews[cc] = len(dct); mpv[cc] = np.mean(list(dct.values()))
    cen = np.full((k, 3), np.nan)
    for cc in range(k):
        pts = P[comp == cc]; pts = pts[~np.isnan(pts[:, 0])]
        if len(pts): cen[cc] = np.median(pts, 0)
    X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r; cand = (nviews >= 3) & ~np.isnan(cen[:, 0]); ring = cand & (np.linalg.norm(X, axis=1) <= 1.0)
    Xr = X[ring]; w_, vv_ = np.linalg.eigh(np.cov(Xr.T)); axis = vv_[:, 1]; t = Xr @ axis; gr = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((gr[:, None] - t[None]) / 0.08) ** 2).sum(1)
    def cut(sign):
        idx = np.where(gr * sign > 0)[0]; idx = idx[np.argsort(np.abs(gr[idx]))]; dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
        for i in mins:
            if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= 0.25 * dens.max(): return float(gr[idx[i]])
        return 1.0 * sign
    cl, cr = cut(-1), cut(1); inside = (t >= cl) & (t <= cr); A_ = cen[ring][inside]; dd, _ = cKDTree(A_).query(A_, k=2); nn = dd[:, 1] / RAD
    absorbed = 100 * np.isin(comp, np.where(ring)[0]).mean()
    line = f"θ {theta} W {window}: {len(ea)} mutual links, {k} components, {int(cand.sum())} with >= 3 views, ring {int(ring.sum())}, row cut {cl:+.2f}/{cr:+.2f} -> apples **{int(inside.sum())}**; views per apple median {np.median(nviews[ring]):.0f} (p10 {np.percentile(nviews[ring], 10):.0f}); masks per view per apple mean {np.mean(mpv[ring]):.2f} (> 1.2 in {100 * np.mean(mpv[ring] > 1.2):.0f} %); NN radii p10 {np.percentile(nn, 10):.1f} / median {np.median(nn):.1f}, closer than 1.5 radii {int((nn < 1.5).sum())}; masks in apples {absorbed:.0f} %"
    if dump:
        np.savez_compressed(R / f'mutual_{a.tag}.npz', comp=comp, nviews=nviews, cen=cen, ring=ring, theta=theta, window=window); json.dump({'tag': a.tag, 'theta': theta, 'window': window, 'apples': int(inside.sum()), 'apples_ring': int(ring.sum()), 'gt': a.gt or None}, open(R / f'registry_mutual_{a.tag}.json', 'w'))
    return line, int(inside.sum())
print(f'[mutual] {a.tree} ({a.tag}): {nM} masks over {V} views', flush=True)
for W_ in (2, 4):
    for th in (0.1, 0.2, 0.3, 0.5):
        line, n = count(th, W_); print('[mutual]   ' + line, flush=True)
line, n = count(a.theta, a.window, dump=True); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f'[mutual] {a.tree} ({a.tag}) chosen ' + line + err, flush=True)
