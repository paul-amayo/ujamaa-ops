#!/usr/bin/env python3
"""smac_eval_closures.py (30 Oct 2026) — score a SMaC teach-and-repeat run against GNSS-anchored truth.
Truth: each survey's tree markers (marker_hierarchy.json, scene frame: y up, ground = x,z; the split transforms are in the z-up LIO frame
with scene (x, z) = (-lio_y, lio_x), checked against the extents of both surveys on 30 Oct 2026) have WGS84 positions (tree_wgs84.json, fitted
from the survey's own GNSS; rmse ~0.4 m), so a 2-D similarity LIO -> ENU is fitted per survey (Umeyama) and both surveys land in one
ENU frame. A repeat keyframe's true position is its first scan's LIO pose carried into ENU; its true reference is the nearest teach
keyframe in ENU. The run's estimated_trajectory_1.txt (one line per repeat keyframe: best closure, its reference, X Y Z yaw of the
closure pose, inliers) is scored at every inlier threshold: a closure is correct if its reference keyframe lies within --tol m of
the true nearest teach keyframe; recall counts repeat keyframes with a correct closure above threshold; the closure pose is also
checked against the true relative pose (translation error in ENU, yaw error incl. the pi ambiguity of opposite driving directions).
Usage: smac_eval_closures.py <run dir> <teach split dir> <repeat split dir> <teach truth prefix> <repeat truth prefix> [--tol 5]
  truth prefix = path prefix of <prefix>_marker_hierarchy.json and <prefix>_tree_wgs84.json"""
import sys, json, csv, argparse, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('teach'); ap.add_argument('repeat'); ap.add_argument('ttruth'); ap.add_argument('rtruth'); ap.add_argument('--tol', type=float, default=5.0); a = ap.parse_args()
R_EARTH = 6378137.0
def enu_fit(prefix, lat0, lon0):
    H = json.load(open(prefix + '_marker_hierarchy.json')); W = json.load(open(prefix + '_tree_wgs84.json'))
    objs = {int(o['id']): np.array(o['xyz'], float) for o in H['objects']}; trees = W['trees']; trees = {int(k): v for k, v in (trees.items() if isinstance(trees, dict) else trees)}
    ids = sorted(set(objs) & set(trees)); A = np.array([[objs[i][0], objs[i][2]] for i in ids]); L = np.array([trees[i] for i in ids], float)
    B = np.c_[(L[:, 1] - lon0) * np.cos(np.radians(lat0)) * np.radians(1) * R_EARTH, (L[:, 0] - lat0) * np.radians(1) * R_EARTH]
    ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A)); d = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, d]) @ Vt
    s = (S * [1, d]).sum() / ((A - ma) ** 2).sum() * len(A); t = mb - s * Rm @ ma; res = np.linalg.norm((s * (Rm @ A.T).T + t) - B, axis=1)
    return (lambda P: s * (Rm @ np.atleast_2d(P).T).T + t), dict(n=len(ids), scale=float(s), rmse=float(np.sqrt((res ** 2).mean()))), Rm
W5 = json.load(open(a.ttruth + '_tree_wgs84.json'))['anchor']; lat0, lon0 = W5['lat0'], W5['lon0']
toENU_t, fit_t, R_t = enu_fit(a.ttruth, lat0, lon0); toENU_r, fit_r, R_r = enu_fit(a.rtruth, lat0, lon0)
print('[truth] teach fit', fit_t, 'repeat fit', fit_r)
def pose(split, i): T = np.loadtxt(Path(split) / 'transforms' / f'transform_{i}.txt'); return T
def lio_to_scene(x, y): return np.array([-y, x])   # z-up LIO ground coordinates -> scene-frame ground coordinates (x, z)
def enu_of(split, i, f): T = pose(split, i); return f(lio_to_scene(T[0, 3], T[1, 3]))[0], T
def yaw_of(T, Rm):   # heading of the lidar x axis on the ground plane (x, z), carried into ENU
    v = lio_to_scene(T[0, 0], T[1, 0]); v = Rm @ v; return float(np.arctan2(v[1], v[0]))
rows = list(csv.DictReader(open(Path(a.run) / 'estimated_trajectory_1.txt')))
k = {c.strip(): c for c in rows[0].keys()}
Q = [dict(q=int(r[k['Query ID']]), ref=int(r[k['Reference ID']]), ref_scan=int(r[k['Ref ScanIdx']]), q_scan=int(r[k['first_scan_index']]), inl=int(r[k['Num Inliers']]), x=float(r[k['X']]), y=float(r[k['Y']]), yaw=float(r[k['Rotation (rad)']])) for r in rows]
# teach keyframes: every reference the run knows (ref_scan per Reference ID), plus all teach keyframes from the ref ids seen
teach_kf = {}
for r in Q: teach_kf.setdefault(r['ref'], r['ref_scan'])
teach_enu = {kid: enu_of(a.teach, si, toENU_t)[0] for kid, si in teach_kf.items()}
TK = np.array(list(teach_enu.keys())); TE = np.array([teach_enu[kid] for kid in TK])
res = []
for r in Q:
    pe, Tq = enu_of(a.repeat, r['q_scan'], toENU_r); d = np.linalg.norm(TE - pe, axis=1); j = int(np.argmin(d)); true_ref = int(TK[j]); true_d = float(d[j])
    ref_e, Tr = enu_of(a.teach, r['ref_scan'], toENU_t); ref_d = float(np.linalg.norm(ref_e - pe))
    yaw_err = abs(((yaw_of(Tq, R_r) - yaw_of(Tr, R_t)) - r['yaw'] + np.pi) % (2 * np.pi) - np.pi)
    res.append(dict(**r, true_ref=true_ref, true_d=true_d, ref_d=ref_d, correct=ref_d <= a.tol, opposite=abs(abs(r['yaw']) - np.pi) < 0.35, yaw_err=float(yaw_err)))
inl = np.array([x['inl'] for x in res]); ok = np.array([x['correct'] for x in res]); n = len(res)
print(f'[run] {n} repeat keyframes with a best closure; reference within {a.tol} m of the truth for {int(ok.sum())} ({100 * ok.mean():.1f} %); median reference offset {np.median([x["ref_d"] for x in res]):.2f} m; opposite-direction closures {int(sum(x["opposite"] for x in res))}')
print('inliers>=  n_closures  precision  recall(of %d)' % n)
for th in (0, 5, 9, 15, 20, 30, 40, 60):
    m = inl >= th; print(f'{th:9d}  {int(m.sum()):10d}  {100 * ok[m].mean() if m.any() else float("nan"):8.1f}%  {100 * (ok & m).sum() / n:6.1f}%')
json.dump(dict(fit_teach=fit_t, fit_repeat=fit_r, tol_m=a.tol, n=n, rows=res), open(Path(a.run) / 'eval_closures.json', 'w'), indent=1); print('[eval] written', Path(a.run) / 'eval_closures.json')
