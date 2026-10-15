#!/usr/bin/env python3
"""fruit_registry_track.py <tree> [tag] [--gate G] [--gt N] — registry by MULTI-VIEW TRACKING-BY-DETECTION (2026-10-15 17:5x, "no shortcuts"):
SAM3 masks are detections, apples are 3-D hypotheses. Views are visited in orbit order (azimuth of the camera about the look-point); in each
view the live hypotheses are projected (the vote projection's viewmat/K), masks are assigned to them by Hungarian matching on the reprojection
distance in units of the mask radius (gate G) with a loose depth gate (|z_hyp - z_lift| <= 0.08 cam_r: an apple behind another is not the
front mask); an assigned mask refines the hypothesis by ray triangulation (least-squares intersection of the assigned rays, 12-14 deg
baselines); an unassigned mask starts a hypothesis at its splat-lifted centroid (masklink_votes_<tag>.npz masks table). Afterwards
hypotheses closer than one apple radius (0.0125 cam_r) with disjoint views are merged (the same apple started twice). Apple = hypothesis
with >= 3 assigned views, inside the camera ring + row cut (masklink_row_cut.py rule). Prints the count-vs-gate curve, views per apple,
nearest-neighbour distances in radii (apples cannot be closer than ~2), and the share of masks assigned. GT-blind; any python with numpy/scipy."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--gate', type=float, default=1.5); ap.add_argument('--zgate', type=float, default=0.08); ap.add_argument('--gt', type=int, default=0); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); v = np.load(R / f'masklink_votes_{a.tag}.npz'); cams = np.load(R / f'cameras_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz')
mk = v['masks']; VM, KK, WH = cams['viewmat'], cams['K'], cams['wh']; V = len(VM); LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); RAD = 0.0125 * cam_r
e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
Cc = np.array([-VM[i, :3, :3].T @ VM[i, :3, 3] for i in range(V)]); az = np.arctan2((Cc - LOOK) @ e2, (Cc - LOOK) @ e1); order = np.argsort(az)
det = {i: [] for i in range(V)}   # per view: (mask idx, u, v, radius_px, lifted xyz, lifted depth)
for row in mk:
    vi, j, x, y, z, area = int(row[0]), int(row[1]), row[2], row[3], row[4], row[5]
    if area <= 0 or not np.isfinite(x): continue
    P = np.array([x, y, z]); pc = VM[vi, :3, :3] @ P + VM[vi, :3, 3]
    if pc[2] <= 0: continue
    u = KK[vi, 0, 0] * pc[0] / pc[2] + KK[vi, 0, 2]; vv = KK[vi, 1, 1] * pc[1] / pc[2] + KK[vi, 1, 2]; det[vi].append((j, u, vv, np.sqrt(area / np.pi), P, pc[2]))
def ray(vi, u, vv):
    d = np.array([(u - KK[vi, 0, 2]) / KK[vi, 0, 0], (vv - KK[vi, 1, 2]) / KK[vi, 1, 1], 1.0]); d = VM[vi, :3, :3].T @ d; return Cc[vi], d / np.linalg.norm(d)
def triangulate(rays, p0):
    if len(rays) < 2: return p0
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c_, d_ in rays: M = np.eye(3) - np.outer(d_, d_); A += M; b += M @ c_
    try: return np.linalg.solve(A + 1e-9 * np.eye(3), b)
    except Exception: return p0
def track(gate):
    hyp = []   # dicts: p, rays, views (set), masks (list of (view, mask))
    for vi in order:
        D = det[vi]
        if not D: continue
        live = [h for h in hyp if True]; cand = []
        for hi, h in enumerate(live):
            pc = VM[vi, :3, :3] @ h['p'] + VM[vi, :3, 3]
            if pc[2] <= 0: continue
            u = KK[vi, 0, 0] * pc[0] / pc[2] + KK[vi, 0, 2]; vv = KK[vi, 1, 1] * pc[1] / pc[2] + KK[vi, 1, 2]
            if -50 <= u < WH[vi, 0] + 50 and -50 <= vv < WH[vi, 1] + 50: cand.append((hi, u, vv, pc[2]))
        assigned = set()
        if cand:
            C = np.full((len(cand), len(D)), 1e6)
            for ci, (hi, u, vv, z) in enumerate(cand):
                for di, (j, du, dv, r, P, zl) in enumerate(D):
                    dist = np.hypot(u - du, vv - dv) / max(r, 1.0)
                    if dist <= gate and abs(z - zl) <= a.zgate * cam_r: C[ci, di] = dist
            ri, cj = linear_sum_assignment(C)
            for r_, c_ in zip(ri, cj):
                if C[r_, c_] >= 1e6: continue
                hi = cand[r_][0]; j, du, dv, r, P, zl = D[c_]; h = live[hi]; h['rays'].append(ray(vi, du, dv)); h['views'].add(vi); h['masks'].append((vi, j)); h['p'] = triangulate(h['rays'], h['p']); assigned.add(c_)
        for di, (j, du, dv, r, P, zl) in enumerate(D):
            if di not in assigned: hyp.append({'p': P.copy(), 'rays': [ray(vi, du, dv)], 'views': {vi}, 'masks': [(vi, j)]})
    # merge duplicates: closer than one radius, disjoint views
    pts = np.array([h['p'] for h in hyp]); tree = cKDTree(pts); pairs = tree.query_pairs(RAD); merged = np.zeros(len(hyp), bool)
    for i_, j_ in sorted(pairs, key=lambda p: -len(hyp[p[0]]['views']) - len(hyp[p[1]]['views'])):
        if merged[i_] or merged[j_] or (hyp[i_]['views'] & hyp[j_]['views']): continue
        hyp[i_]['rays'] += hyp[j_]['rays']; hyp[i_]['views'] |= hyp[j_]['views']; hyp[i_]['masks'] += hyp[j_]['masks']; hyp[i_]['p'] = triangulate(hyp[i_]['rays'], hyp[i_]['p']); merged[j_] = True
    hyp = [h for h, m in zip(hyp, merged) if not m]; return hyp
def count(gate, dump=False):
    hyp = track(gate); nv = np.array([len(h['views']) for h in hyp]); P = np.array([h['p'] for h in hyp]); X = np.stack([(P - LOOK) @ e1, (P - LOOK) @ e2], 1) / cam_r
    cand = nv >= 3; ring = cand & (np.linalg.norm(X, axis=1) <= 1.0); Xr = X[ring]; w_, vv_ = np.linalg.eigh(np.cov(Xr.T)); axis = vv_[:, 1]; t = Xr @ axis; gr = np.arange(-1, 1.001, 0.01); dens = np.exp(-0.5 * ((gr[:, None] - t[None]) / 0.08) ** 2).sum(1)
    def cut(sign):
        idx = np.where(gr * sign > 0)[0]; idx = idx[np.argsort(np.abs(gr[idx]))]; dd = dens[idx]; mins = [i for i in range(1, len(dd) - 1) if dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]]
        for i in mins:
            if dd[i + 1:].max() >= 1.5 * dd[i] and dd[i + 1:].max() >= 0.25 * dens.max(): return float(gr[idx[i]])
        return 1.0 * sign
    cl, cr = cut(-1), cut(1); inside = (t >= cl) & (t <= cr); A_ = P[ring][inside]; dd, _ = cKDTree(A_).query(A_, k=2); nn = dd[:, 1] / RAD
    nmask = sum(len(d) for d in det.values()); used = sum(len(h['masks']) for h in hyp if len(h['views']) >= 3)
    line = f"gate {gate}: {len(hyp)} hypotheses, {int(cand.sum())} with >= 3 views, ring {int(ring.sum())}, row cut {cl:+.2f}/{cr:+.2f} -> apples **{int(inside.sum())}**; views per apple median {np.median(nv[ring]):.0f} (p10 {np.percentile(nv[ring], 10):.0f}); NN radii p10 {np.percentile(nn, 10):.1f} / median {np.median(nn):.1f}, closer than 1.5 radii {int((nn < 1.5).sum())}; masks in >= 3-view apples {100 * used / nmask:.0f} %"
    if dump:
        np.savez_compressed(R / f'track_{a.tag}.npz', p=P, nviews=nv, ring=ring, gate=gate, masks=np.array([(k, vi_, j_) for k, h in enumerate(hyp) for vi_, j_ in h['masks']]))
        json.dump({'tag': a.tag, 'gate': gate, 'zgate': a.zgate, 'apples': int(inside.sum()), 'apples_ring': int(ring.sum()), 'gt': a.gt or None}, open(R / f'registry_track_{a.tag}.json', 'w'))
    return line, int(inside.sum())
print(f"[track] {a.tree} ({a.tag}): {V} views in orbit order, {sum(len(d) for d in det.values())} mask detections with a lifted centroid", flush=True)
for g_ in (1.0, 1.5, 2.0, 3.0):
    line, n = count(g_); print('[track]   ' + line, flush=True)
line, n = count(a.gate, dump=True); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f'[track] {a.tree} ({a.tag}) chosen ' + line + err, flush=True)
