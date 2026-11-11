#!/usr/bin/env python3
"""azalai_online_geom.py (A100 agent, 11 Nov 2026): lane passes, inner rows and lane frames of the RR design, rebuilt from the
design text (the same code as in azalai_online_resim.py, which keeps its own copy as the independent check). Imported by
azalai_online_datum.py and azalai_online_panels.py."""
import json
from pathlib import Path
import numpy as np

X = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental')
SK = X / 'sankofa/jirs_20261023/lidar_fixedreader'
EXP = json.load(open(X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'))['experiences']


def lane_passes(c2w):
    n = len(c2w); fin = np.isfinite(c2w.reshape(n, -1)).all(1); zc = c2w[:, 2, 2]
    sense = np.where(fin & (zc > 0.8), 1, np.where(fin & (zc < -0.8), -1, 0)); runs, start = [], None
    for k in range(n):
        if sense[k] == 0:
            if start is not None: runs.append((start, k - 1)); start = None
            continue
        if start is None: start = k
        elif sense[k] != sense[start] or abs(c2w[k, 0, 3] - c2w[k - 1, 0, 3]) > 0.6: runs.append((start, k - 1)); start = k
    if start is not None: runs.append((start, n - 1))
    return [(a, b, int(sense[a])) for a, b in runs if b - a + 1 >= 30]


_cache = {}
def load_pass(name):
    if name in _cache: return _cache[name]
    c2w = np.array(EXP[name]['c2w'], float); z = np.load(SK / f'census_{name}.npz')
    B = z['bags_xyz'].astype(float); ob, ok_, on = z['obs_bag'].astype(int), z['obs_k'].astype(int), z['obs_n'].astype(int); lanes = []
    for k0, k1, s in lane_passes(c2w):
        P = c2w[k0:k1 + 1, :3, 3][:, [0, 2]]; m = P.mean(0); w, V = np.linalg.eigh(np.cov((P - m).T)); d = V[:, np.argmax(w)]; d = d if d[1] > 0 else -d
        seen = np.unique(ob[(ok_ >= k0) & (ok_ <= k1)]); seen = seen[np.isfinite(B[seen]).all(1)]; lat = (B[seen][:, [0, 2]] - m) @ np.array([d[1], -d[0]])
        keep = np.abs(lat) <= 1.5; ids, lat = seen[keep], lat[keep]; left = lat < 0; lane = dict(k0=k0, k1=k1, sense=s, ids=ids, left=left, frame=None)
        if left.sum() >= 2 and (~left).sum() >= 2:
            Q = B[ids]; c = Q.mean(0); _, V3 = np.linalg.eigh(np.cov((Q - c).T)); nrm = V3[:, 0]; nrm = nrm if nrm[1] < 0 else -nrm
            ez = np.array([0.0, 0.0, 1.0]); ea = ez - nrm * (nrm @ ez); ea /= np.linalg.norm(ea); el = np.cross(nrm, ea)
            slope, cl, cr = np.linalg.lstsq(np.column_stack([(Q - c) @ ea, left.astype(float), (~left).astype(float)]), (Q - c) @ el, rcond=None)[0]
            axis = ea + slope * el; axis /= np.linalg.norm(axis); mid = c + 0.5 * (cl + cr) * el; st = (Q - mid) @ axis
            lane['frame'] = (mid + (st.min() if s > 0 else st.max()) * axis, axis); lane['sig'] = s * ((Q - lane['frame'][0]) @ axis)
        lanes.append(lane)
    out = dict(c2w=c2w, B=B, lanes=lanes, ob=ob, ok=ok_, on=on); _cache[name] = out; return out


def dstation(lane, p):
    o, a = lane['frame']; return float(lane['sense'] * ((np.asarray(p, float) - o) @ a))


def placed(name):
    return np.eye(4) if name == 'dec' else np.load(SK / f'placed_{name}.npz')['G_global'].astype(float)


