#!/usr/bin/env python3
"""smac_eval_descriptors.py (30 Oct 2026) — score the SegMatch descriptor arm (SegMatch segments of each repeat keyframe matched by
eigenvalue descriptors + geometric consistency against the teach map as target cloud) with the same truth as smac_eval_closures.py.
A recognised match pairs a source segment (centroid in the keyframe frame, the pose of the scan before the keyframe's first scan) with a
target segment (centroid in the teach LIO frame). Both centroids are carried into ENU; a match is correct if they lie within --tol m.
A repeat keyframe is localised if it has at least one recognised match, correctly if the majority of its matches are correct.
Usage: smac_eval_descriptors.py <run dir> <teach split> <repeat split> <teach truth prefix> <repeat truth prefix> [--tol 5]"""
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
    s = (S * [1, d]).sum() / ((A - ma) ** 2).sum() * len(A); t = mb - s * Rm @ ma
    return lambda P: s * (Rm @ np.atleast_2d(P).T).T + t
W5 = json.load(open(a.ttruth + '_tree_wgs84.json'))['anchor']; lat0, lon0 = W5['lat0'], W5['lon0']
f_t = enu_fit(a.ttruth, lat0, lon0); f_r = enu_fit(a.rtruth, lat0, lon0)
def scene(xy): return np.c_[-xy[:, 1], xy[:, 0]]           # z-up LIO ground (x, y) -> scene ground (x, z)
def pose(split, i): return np.loadtxt(Path(split) / 'transforms' / f'transform_{i}.txt')
kf = {int(r[0]): dict(session=r[1], first=int(r[2]), last=int(r[3])) for r in csv.reader(open(Path(a.run) / 'keyframes.csv'))}
rows = [r for r in csv.reader(open(Path(a.run) / 'matching_centroids.csv')) if len(r) >= 9]
per = {}
for r in rows:
    k = int(r[0]) // 1000000000
    if k not in kf or kf[k]['session'] != 'repeat': continue
    src = np.array(r[3:6], float); tgt = np.array(r[6:9], float)
    o = kf[k]['first'] - 1; T = pose(a.repeat, o) if o >= 1 else np.eye(4)
    src_lio = T[:3, :3] @ src + T[:3, 3]
    e_src = f_r(scene(src_lio[None, :2]))[0]; e_tgt = f_t(scene(tgt[None, :2]))[0]
    per.setdefault(k, []).append(float(np.linalg.norm(e_src - e_tgt)))
rep = sorted(k for k, v in kf.items() if v['session'] == 'repeat'); n = len(rep)
loc = [k for k in rep if k in per]; good = [k for k in loc if np.mean(np.array(per[k]) <= a.tol) > 0.5]
alld = np.concatenate([per[k] for k in loc]) if loc else np.array([])
print(f'[descriptor arm] {n} repeat keyframes; {len(loc)} with recognised matches ({100 * len(loc) / max(n, 1):.1f} %); {len(good)} localised correctly (majority of matches within {a.tol} m) = {100 * len(good) / max(n, 1):.1f} % of keyframes, precision {100 * len(good) / max(len(loc), 1):.1f} %')
if len(alld): print(f'  {len(alld)} recognised matches, {int((alld <= a.tol).sum())} within {a.tol} m ({100 * (alld <= a.tol).mean():.1f} %); median centroid offset {np.median(alld):.1f} m')
json.dump(dict(tol_m=a.tol, n_repeat_keyframes=n, localised=len(loc), correct=len(good), per_keyframe={k: per[k] for k in loc}), open(Path(a.run) / 'eval_descriptors.json', 'w'), indent=1); print('[eval] written', Path(a.run) / 'eval_descriptors.json')
