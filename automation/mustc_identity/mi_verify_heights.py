#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3), A100 session 11-11: independent re-derivation of measurement (ii). Shares no code with mi_heights.py.
Differences of path, on purpose:
  * the source camera of a point is taken from transforms_ba.json (rig metres, OpenGL c2w turned to OpenCV), not from cameras.json
    (model units through the dataparser transform), and the point is brought to the rig by the inverse of T_rig_to_utm_datum_corrected;
  * the soil polygon test is matplotlib.path, not a ray-casting loop; the h95 is np.quantile; the mutual nearest neighbour is a brute-force
    distance matrix, not cKDTree; c2 is re-read from the record.
Reads heights.csv and heights_summary.json only at the end, to compare.
nerf_new pixi env:  mi_verify_heights.py --stop 230613_pos00 --dir <stop dir> [--out verify_heights.json]"""
import argparse, csv, json
import numpy as np
from matplotlib.path import Path as MPath
ap = argparse.ArgumentParser(); ap.add_argument('--stop', required=True); ap.add_argument('--dir', required=True); ap.add_argument('--out', default='')
ap.add_argument('--census', default='/mnt/shared2/must-c/plot198/census'); ap.add_argument('--geom', default='/mnt/shared/must-c_bulk/work/splat_geom')
ap.add_argument('--ws', default='/mnt/shared/must-c_bulk/work/ws'); ap.add_argument('--pred', default='pred_tangent_vf')
A = ap.parse_args(); date = A.stop.split('_')[0]
z = np.load(f'{A.geom}/{A.stop}/depth_points_utm.npz')
xyz, hh, keep, src = z['xyz'], z['h'].astype(np.float64), z['keep'].astype(bool), z['src_cam'].astype(np.int64)
poly = MPath(np.asarray(z['footprint_soil_polygon_utm'])); inside = poly.contains_points(xyz[:, :2]); use = keep & inside
Td = np.asarray(z['T_rig_to_utm_datum_corrected'], np.float64); rig = np.linalg.solve(Td, np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
TR = json.load(open(f'{A.ws}/{A.stop}/transforms_ba.json')); train = set(json.load(open(f'{A.ws}/{A.stop}/split.json'))['train'])
frs = [f for f in TR['frames'] if f['camera_label'] in train]
addr = np.full(len(xyz), -9, np.int64); off_lattice = []
for k, f in enumerate(frs):
    s = src == k
    if not s.any(): continue
    c2w = np.asarray(f['transform_matrix'], np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])     # OpenGL -> OpenCV camera axes
    pc = (np.linalg.inv(c2w) @ np.c_[rig[s], np.ones(s.sum())].T).T
    u = f['fl_x'] * pc[:, 0] / pc[:, 2] + f['cx']; v = f['fl_y'] * pc[:, 1] / pc[:, 2] + f['cy']
    off_lattice.append(np.abs(np.c_[(u - 0.5) / 2 - np.rint((u - 0.5) / 2), (v - 0.5) / 2 - np.rint((v - 0.5) / 2)]).max(1) * 2)
    name = f['file_path'].split('/')[-1]; t = np.load(f'{A.dir}/{A.pred}/{name}.npz')['tree']
    col = (u * t.shape[1] / f['w']).astype(np.int64); row = (v * t.shape[0] / f['h']).astype(np.int64)
    addr[s] = t[np.clip(row, 0, t.shape[0] - 1), np.clip(col, 0, t.shape[1] - 1)]
off = np.concatenate(off_lattice)
plants = {int(p['id']): p for p in json.load(open(f'{A.census}/lmi_{date}.json'))['plants']}
P = {int(p['id']): p for p in json.load(open(f'{A.dir}/plants.json'))['plants']}
S = sorted(i for i, p in P.items() if p['on_row'] and p['in_soil_polygon'])
mine = {}
for i in S:
    m = use & (addr == i); n = int(m.sum())
    mine[i] = {'n': n, 'h95': float(np.quantile(hh[m], 0.95)) if n >= 20 else None}
# c1 by brute force, ledger band, no stakes on the date (re-read)
SC = json.load(open(f'{A.census}/splat_census/{A.stop}__b.json'))['instances']; sxy = np.array([[q['x_utm'], q['y_utm']] for q in SC])
band = lambda p: 357415.6 <= p['x'] <= 357421.7 and 5610200.3 <= p['y'] <= 5610201.95
stk = set(json.load(open(f'{A.census}/lmi_assoc.json'))['stake_rule']['per_date'].get(date, {}).get('ids', []))
lids = [i for i in sorted(P) if band(plants[i]) and i not in stk]; lxy = np.array([P[i]['xyz_ref'][:2] for i in lids])
def mnn(a, b, r):
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2); out = []
    for ia in range(len(a)):
        jb = int(np.argmin(D[ia]))
        if D[ia, jb] <= r and int(np.argmin(D[:, jb])) == ia: out.append((ia, jb, float(D[ia, jb])))
    return out
p1 = mnn(sxy, lxy, 0.10); tr = None
if len(p1) >= 5:
    tr = np.median(np.array([lxy[j] - sxy[i] for i, j, _ in p1]), axis=0); p1 = mnn(sxy + tr, lxy, 0.10)
c1 = {lids[j]: float(SC[i]['h95_m']) for i, j, _ in p1}
c2 = {}   # the record numbers its plants locally: join to the lmi id by the nearest raw laser position (must be under 1 mm)
for q in json.load(open(f'{A.census}/dsm_vs_laser.json'))[A.stop]['vs_laser']['by_footprint']['nview6']['b']['per_plant']['plants']:
    best = min(plants, key=lambda i: (plants[i]['x'] - q['x']) ** 2 + (plants[i]['y'] - q['y']) ** 2)
    if ((plants[best]['x'] - q['x']) ** 2 + (plants[best]['y'] - q['y']) ** 2) ** 0.5 < 1e-3: c2[best] = q.get('splat_h95')
# compare with mi_heights.py's outputs
rows = {int(r['plant']): r for r in csv.DictReader(open(f'{A.dir}/heights.csv'))}
fl = lambda x: None if x in ('', None) else round(float(x), 4)
diff = []
for i in S:
    r = rows.get(i, {})
    for key, val in (('address_h95_m', mine[i]['h95']), ('c1_h95_m', c1.get(i)), ('c2_h95_m', c2.get(i))):
        a_, b_ = fl(r.get(key)), (None if val is None else round(float(val), 4))
        if a_ != b_: diff.append({'plant': i, 'field': key, 'heights_csv': a_, 'verify': b_})
    if int(r.get('address_n_points', -1)) != mine[i]['n']: diff.append({'plant': i, 'field': 'address_n_points', 'heights_csv': r.get('address_n_points'), 'verify': mine[i]['n']})
def stat(vals):
    d = np.array([v - plants[i]['height_p95'] for i, v in vals.items() if v is not None])
    return {'n': int(len(d))} if not len(d) else {'n': int(len(d)), 'bias_median_m': round(float(np.median(d)), 4), 'absdiff_median_m': round(float(np.median(np.abs(d))), 4), 'absdiff_p95_m': round(float(np.percentile(np.abs(d), 95)), 4)}
res = {'S': S, 'points_used': int(use.sum()), 'points_used_with_address': int((addr[use] >= 0).sum()),
       'pixel_recovery_off_lattice_px': {'median': float(np.median(off)), 'p99': float(np.percentile(off, 99)), 'max': float(off.max())},
       'address': stat({i: mine[i]['h95'] for i in S}), 'c1': stat({i: c1.get(i) for i in S}), 'c2': stat({i: c2.get(i) for i in S}),
       'c1_translation_en_m': None if tr is None else [round(float(t), 5) for t in tr], 'per_plant': mine, 'differences_vs_heights_csv': diff}
print(json.dumps({k: v for k, v in res.items() if k != 'per_plant'}, indent=1))
if A.out: json.dump(res, open(A.out, 'w'), indent=1)
