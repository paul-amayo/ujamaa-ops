#!/usr/bin/env python3
"""fruit_registry_community.py <tree> [tag] [--res R] [--gt N] — stage 2b of the mask-graph registry (2026-10-15 17:3x): apples as COMMUNITIES of
gaussians. Nodes = gaussians that are top contributors (>= 1 % of the mask mass) to >= 2 SAM3 masks; edge weight = cosine similarity of
their mask-mass profiles (which masks they paint, how much), top-30 edges per node; Louvain (igraph community_multilevel, resolution R).
Touching apples are dense communities joined by thin bridges (boundary gaussians) — what modularity separates and single linkage cannot.
A community OWNS a mask when its members hold the majority of the mask's kept mass; apple = community owning masks in >= 3 views, placed at
the mass-weighted mean of its members (means_<tag>.npy from the checkpoint), inside the camera ring + row cut (as masklink_row_cut.py).
Diagnostics: owned masks per view per community (1 = clean instance), masks owned by any apple, apple pairs closer than one apple radius.
GT-blind; the count-vs-resolution curve is printed. venv: /home/paperspace/venvs/leiden (python-igraph)."""
import argparse, json
from pathlib import Path
import numpy as np, igraph as ig
from scipy.sparse import csr_matrix
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--res', type=float, default=1.0); ap.add_argument('--gt', type=int, default=0); ap.add_argument('--topw', type=float, default=0.01); ap.add_argument('--knn', type=int, default=30); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); d = np.load(R / f'maskgraph_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz'); means = np.load(R / f'means_{a.tag}.npy')
mrow, gid, w, mview, mmass = d['mrow'], d['gid'], d['w'], d['mview'], d['mmass']; nM = len(mview); N = int(d['N'])
keep = w >= a.topw * mmass[mrow]; G = csr_matrix((w[keep], (gid[keep], mrow[keep])), shape=(N, nM)); deg = np.diff(G.indptr); nodes = np.where(deg >= 2)[0]; Gs = G[nodes]
nrm = np.sqrt(np.asarray(Gs.multiply(Gs).sum(1)).ravel()); nrm[nrm == 0] = 1; Gn = csr_matrix(Gs.multiply(1 / nrm[:, None])); S = (Gn @ Gn.T).tocsr(); S.setdiag(0); S.eliminate_zeros()
# top-k edges per node
rows, cols, vals = [], [], []
for i in range(S.shape[0]):
    s0, e0 = S.indptr[i], S.indptr[i + 1]
    if e0 - s0 == 0: continue
    cc, vv = S.indices[s0:e0], S.data[s0:e0]; o = np.argsort(-vv)[:a.knn]; rows.append(np.full(len(o), i)); cols.append(cc[o]); vals.append(vv[o])
rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals); m_ = rows < cols; pairs = {}
for r_, c_, v_ in zip(rows, cols, vals): pairs[(min(r_, c_), max(r_, c_))] = max(v_, pairs.get((min(r_, c_), max(r_, c_)), 0))
E = np.array(list(pairs.keys())); Wt = np.array(list(pairs.values())); g = ig.Graph(n=len(nodes), edges=E.tolist()); g.es['weight'] = Wt.tolist()
print(f'[community] {a.tree} ({a.tag}): {nM} masks, {len(nodes)} gaussians painting >= 2 masks, {len(E)} edges (top-{a.knn}, cos >= 0)', flush=True)
LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
mass_node = np.asarray(Gs.sum(1)).ravel(); Gcsc = Gs.tocsc()
def run(res, dump=False):
    part = g.community_multilevel(weights='weight', resolution=res); lab = np.array(part.membership); k = lab.max() + 1
    # mask ownership: majority of the kept mass
    own = np.full(nM, -1); 
    for j in range(nM):
        s0, e0 = Gcsc.indptr[j], Gcsc.indptr[j + 1]
        if e0 == s0: continue
        li = lab[Gcsc.indices[s0:e0]]; ww = Gcsc.data[s0:e0]; tot = ww.sum(); acc = np.bincount(li, weights=ww, minlength=k); b = int(acc.argmax())
        if acc[b] > 0.5 * tot: own[j] = b
    owned = own >= 0; nviews = np.zeros(k, int); mpv = np.zeros(k); 
    for b in range(k):
        vs = mview[owned & (own == b)]
        if len(vs): u, cnt = np.unique(vs, return_counts=True); nviews[b] = len(u); mpv[b] = np.median(cnt)
    cen = np.zeros((k, 3))
    for q in range(3): cen[:, q] = np.bincount(lab, weights=means[nodes, q] * mass_node, minlength=k) / np.maximum(np.bincount(lab, weights=mass_node, minlength=k), 1e-9)
    ext = np.sqrt(np.bincount(lab, weights=mass_node * ((means[nodes] - cen[lab]) ** 2).sum(1), minlength=k) / np.maximum(np.bincount(lab, weights=mass_node, minlength=k), 1e-9)) / (0.0125 * cam_r)   # mass-weighted RMS extent in apple radii (a sphere of radius r has RMS 0.77 r)
    X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r; cand = nviews >= 3; ring = cand & (np.linalg.norm(X, axis=1) <= 1.0)
    Xr = X[ring]; wv, vv = np.linalg.eigh(np.cov(Xr.T)); axis = vv[:, 1]; t = Xr @ axis; gr = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((gr[:, None] - t[None]) / 0.08) ** 2).sum(1)
    def cut(sign):
        idx = np.where(gr * sign > 0)[0]; idx = idx[np.argsort(np.abs(gr[idx]))]; dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
        for i in mins:
            if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= 0.25 * dens.max(): return float(gr[idx[i]])
        return 1.0 * sign
    cl, cr = cut(-1), cut(1); inside = (t >= cl) & (t <= cr); n_ring, n_row = int(ring.sum()), int(inside.sum())
    from scipy.spatial import cKDTree
    A_ = cen[ring][inside]; dd, _ = cKDTree(A_).query(A_, k=2); nn = dd[:, 1] / (0.0125 * cam_r); close = int((nn < 1.5).sum()) if len(A_) > 1 else 0   # apple radius 0.0125 cam_r (measured RMS extents); apples cannot be closer than ~2 radii
    line = f"res {res}: {k} communities, {int(cand.sum())} with >= 3 views, ring {n_ring}, row cut {cl:+.2f}/{cr:+.2f} -> apples **{n_row}**; views per apple median {np.median(nviews[ring]) if n_ring else 0:.0f}; owned masks per view per apple median {np.median(mpv[ring]) if n_ring else 0:.1f}; masks owned by an apple {100 * np.isin(own, np.where(ring)[0]).mean():.0f} %; NN distance in radii p10 {np.percentile(nn, 10):.1f} / p25 {np.percentile(nn, 25):.1f} / median {np.median(nn):.1f}, apples closer than 1.5 radii to another {close} ({100 * close / max(1, len(A_)):.0f} %); extent in radii p25 {np.percentile(ext[ring], 25):.2f} / median {np.median(ext[ring]):.2f} / p75 {np.percentile(ext[ring], 75):.2f}, > 1.2 (merged) {100 * np.mean(ext[ring] > 1.2):.0f} %, < 0.4 (fragment) {100 * np.mean(ext[ring] < 0.4):.0f} %"
    if dump: np.savez_compressed(R / f'community_{a.tag}.npz', nodes=nodes, lab=lab, own=own, cen=cen, ring=ring, nviews=nviews, res=res); json.dump({'tag': a.tag, 'res': res, 'apples': n_row, 'apples_ring': n_ring, 'gt': a.gt or None}, open(R / f'registry_community_{a.tag}.json', 'w'))
    return line, n_row
for res in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
    line, n = run(res); print('[community]   ' + line, flush=True)
line, n = run(a.res, dump=True); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f'[community] {a.tree} ({a.tag}) chosen ' + line + err, flush=True)
