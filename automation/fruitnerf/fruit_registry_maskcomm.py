#!/usr/bin/env python3
"""fruit_registry_maskcomm.py <tree> [tag] [--res R] [--min-views M] [--gt N] — registry stage 2c (2026-10-15 18:5x): MASK COMMUNITIES + the
view constraint. Measured today: (a) the splat renders an apple consistently — same-apple masks in adjacent views have cosine 0.91 of their
rendering-gaussian mass vectors, different apples 0.00; (b) single linkage on that similarity either fragments apples (high threshold) or
chains the canopy (low threshold); (c) Louvain communities of the MASK graph keep clean apples whole (95 % of an apple's masks in one
community at every resolution) but touching apples share a community (several masks per view). So: Louvain (igraph, resolution R) on the
mask graph (edges cos >= 0.1), then inside each community a greedy agglomeration by descending cosine under CANNOT-LINKS = two masks of the
same view that share no gaussians (cos < 0.1 — different apples in that view); same-view pairs that do share gaussians (pieces, nested
duplicates) are free to merge. Apple = sub-cluster with >= M views, placed at the median lifted centroid, inside the camera ring + row cut.
Diagnostics: masks per view per apple (merge indicator), NN distance in radii (duplicate indicator), clean record apples' fragmentation.
venv /home/paperspace/venvs/leiden (python-igraph)."""
import argparse, json
from pathlib import Path
import numpy as np, igraph as ig
from collections import defaultdict
from scipy.sparse import csr_matrix, triu
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--res', type=float, default=1.0); ap.add_argument('--min-views', type=int, default=5); ap.add_argument('--gt', type=int, default=0); ap.add_argument('--cl', type=float, default=0.1); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); v = np.load(R / f'masklink_votes_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz'); d = np.load(R / f'maskgraph_{a.tag}.npz')
comp0, apple0 = c['comp'], c['apple']; vi, lab, row = v['vi'].astype(int), v['lab'].astype(int), v['row'].astype(int); cv = comp0[row]
mrow, gid, w, mview, mlab = d['mrow'], d['gid'], d['w'], d['mview'], d['mlab']; nM = len(mview); N = int(d['N'])
LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); RAD = 0.0125 * cam_r; e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
C = csr_matrix((w, (mrow, gid)), shape=(nM, N)); nrm = np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel()); nrm[nrm == 0] = 1; Cn = csr_matrix(C.multiply(1 / nrm[:, None])); Of = (Cn @ Cn.T).tocsr(); O = triu(Of.tocoo(), k=1).tocoo(); midx = {(int(a_), int(b_)): i for i, (a_, b_) in enumerate(zip(mview, mlab))}
mk = v['masks']; pos = {(int(r[0]), int(r[1])): r[2:5] for r in mk}; P = np.array([pos.get((int(vv), int(ll)), [np.nan] * 3) for vv, ll in zip(mview, mlab)])
votes = defaultdict(lambda: defaultdict(int))
for a_, vi_, l_ in zip(cv, vi, lab):
    if apple0[a_]: votes[a_][(vi_, l_)] += 1
clean = []
for a_, vm_ in votes.items():
    best = {}
    for (vi_, l_), n in vm_.items():
        if n > best.get(vi_, (0, None))[0]: best[vi_] = (n, l_)
    per_view = defaultdict(set)
    for (vi_, l_), n in vm_.items():
        if n >= 2: per_view[vi_].add(l_)
    if np.median([len(s) for s in per_view.values()]) > 1 or len(best) < 10: continue
    clean.append([midx[(vi_, l_)] for vi_, (n, l_) in best.items() if (vi_, l_) in midx])
keep = O.data >= 0.1; g = ig.Graph(n=nM, edges=np.stack([O.row[keep], O.col[keep]], 1).tolist()); g.es['weight'] = O.data[keep].tolist()
def split_community(members):
    """greedy agglomeration by descending cosine under same-view cannot-links (cos < cl)."""
    if len(members) == 1: return [members]
    sub = Of[members][:, members].toarray(); iu = np.triu_indices(len(members), 1); s_ = sub[iu]; o = np.argsort(-s_); par = np.arange(len(members)); byview = [{int(mview[m]): [i] for i, m in [(i, m)]} for i, m in enumerate(members)]
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for t in o:
        if s_[t] < 0.1: break
        i, j = iu[0][t], iu[1][t]; ri, rj = f(i), f(j)
        if ri == rj: continue
        ok = True
        for vv, li in byview[ri].items():
            lj = byview[rj].get(vv)
            if lj and sub[np.ix_(li, lj)].min() < a.cl: ok = False; break
        if not ok: continue
        if len(byview[ri]) < len(byview[rj]): ri, rj = rj, ri
        par[rj] = ri
        for vv, lj in byview[rj].items(): byview[ri].setdefault(vv, []).extend(lj)
        byview[rj] = {}
    roots = np.array([f(i) for i in range(len(members))]); return [members[roots == r_] for r_ in np.unique(roots)]
def run(res, dump=False):
    part = g.community_multilevel(weights='weight', resolution=res); cm = np.array(part.membership); comp = np.full(nM, -1); nxt = 0; nsplit = 0
    for cc in range(cm.max() + 1):
        members = np.where(cm == cc)[0]; parts = split_community(members); nsplit += len(parts) > 1
        for pt in parts: comp[pt] = nxt; nxt += 1
    k = nxt; nv = np.array([len(set(mview[comp == cc].tolist())) for cc in range(k)]); nm = np.bincount(comp, minlength=k); mpv = nm / np.maximum(nv, 1)
    cen = np.full((k, 3), np.nan)
    for cc in range(k):
        pts = P[comp == cc]; pts = pts[~np.isnan(pts[:, 0])]
        if len(pts): cen[cc] = np.median(pts, 0)
    X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r; ok = ~np.isnan(cen[:, 0]); cand = ok & (nv >= a.min_views); ring = cand & (np.linalg.norm(np.nan_to_num(X), axis=1) <= 1.0)
    Xr = X[ring]; w_, vv_ = np.linalg.eigh(np.cov(Xr.T)); axis = vv_[:, 1]; t = Xr @ axis; gr = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((gr[:, None] - t[None]) / 0.08) ** 2).sum(1)
    def cut(sign):
        idx = np.where(gr * sign > 0)[0]; idx = idx[np.argsort(np.abs(gr[idx]))]; dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
        for i in mins:
            if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= 0.25 * dens.max(): return float(gr[idx[i]])
        return 1.0 * sign
    cl_, cr_ = cut(-1), cut(1); inside = (t >= cl_) & (t <= cr_); A_ = cen[ring][inside]; dd, _ = cKDTree(A_).query(A_, k=2); nn = dd[:, 1] / RAD
    per_apple = [len(set(comp[ms])) for ms in clean]; whole = np.mean([max(np.bincount(comp[ms])) / len(ms) for ms in clean]) if clean else 0
    n = int(inside.sum()); line = f"res {res}: {cm.max() + 1} communities, {nsplit} split by the view constraint -> {k} clusters; apples (>= {a.min_views} views, ring {int(ring.sum())}, row cut {cl_:+.2f}/{cr_:+.2f}) **{n}**; views per apple median {np.median(nv[ring]):.0f} (p10 {np.percentile(nv[ring], 10):.0f}); masks per view > 1.5 in {int((mpv[ring] > 1.5).sum())}; NN radii p10 {np.percentile(nn, 10):.1f} / median {np.median(nn):.1f}, closer than 1.5 radii {int((nn < 1.5).sum())}; clean apples: {np.median(per_apple):.0f} clusters (p75 {np.percentile(per_apple, 75):.0f}), largest part {100 * whole:.0f} %"
    if dump: np.savez_compressed(R / f'maskcomm_{a.tag}.npz', comp=comp, nviews=nv, cen=cen, ring=ring, res=res, min_views=a.min_views); json.dump({'tag': a.tag, 'res': res, 'min_views': a.min_views, 'apples': n, 'apples_ring': int(ring.sum()), 'gt': a.gt or None}, open(R / f'registry_maskcomm_{a.tag}.json', 'w'))
    return line, n
print(f"[maskcomm] {a.tree} ({a.tag}): {nM} masks, {int(keep.sum())} edges, {len(clean)} clean record apples for the fragmentation check", flush=True)
for res in (0.5, 1.0, 2.0, 4.0):
    line, n = run(res); print('[maskcomm]   ' + line, flush=True)
line, n = run(a.res, dump=True); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f'[maskcomm] {a.tree} ({a.tag}) chosen ' + line + err, flush=True)
