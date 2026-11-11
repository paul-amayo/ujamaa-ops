#!/usr/bin/env python3
"""azalai_rr_rowend_posthoc.py (11 Nov 2026, A100 session) — POST HOC, NOT A RULE: for every datum failure of the geometry-only pairs
(azalai_rr_geom.py outputs), why the two passes' first inner-row bags differ: each pass's first bag is looked up in the other census
(nearest bag within 0.5 m in the December ground plane after the site-wide fits); if there is one, is it in the other pass's inner
rows, was it seen from that pass's lane keyframes, and how far is it from that pass's camera line (the 1.5 m threshold).
Writes rr_rowend_posthoc.json. nerf_new python3.10:  azalai_rr_rowend_posthoc.py <out.json>"""
import sys, json, numpy as np
from scipy.spatial import cKDTree
SK = '/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/sankofa/jirs_20261023/lidar_fixedreader/'
E = json.load(open('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'))['experiences']
def placed(n): return np.load(SK + 'census_dec.npz')['bags_xyz'].astype(float) if n == 'dec' else np.load(SK + f'placed_{n}.npz')['global'].astype(float)
def own(n): return np.load(SK + f'census_{n}.npz')
def seen_by(n, k0, k1): z = own(n); return {int(b) for b, k in zip(z['obs_bag'], z['obs_k']) if k0 <= k <= k1}
def lat_of(n, k0, k1, p):
    C = np.array(E[n]['c2w'])[k0:k1 + 1, :3, 3]; m = C[:, [0, 2]].mean(0); d = np.linalg.svd(C[:, [0, 2]] - m)[2][0]; d = d * np.sign(d[1]); nrm = np.array([d[1], -d[0]]); return float((p[[0, 2]] - m) @ nrm)
OUT = {}
for s in ('dec_apr1130_allmnn', 'dec_apr1400_allmnn', 'dec_apr16_1230_allmnn', 'apr1130_apr1400', 'apr1130_apr16_1230'):
    J = json.load(open(f'/mnt/shared/azalai_rr/geom/{s}/rr_geom_{s}.json')); S = J['summary']; ra, rb = S['registry'], S['revisit']; A = placed(ra); B = placed(rb)
    Aown = own(ra)['bags_xyz'].astype(float); Bown = own(rb)['bags_xyz'].astype(float); ia = np.flatnonzero(np.isfinite(A).all(1)); ib = np.flatnonzero(np.isfinite(B).all(1)); ta = cKDTree(A[ia][:, [0, 2]]); tb = cKDTree(B[ib][:, [0, 2]])
    rows = []
    for l in J['lanes_full']:
        if not l.get('datum_failure'): continue
        i = l['lane']; pa = J['twin_passes'][i - 1]; pb = J['rev_passes'][i - 1]; inA = set(J['twin_inner'][str(i)]['bags']); inB = set(J['rev_inner'][str(i)]['bags']); r = dict(lane=i, offset_m=l['row_end_minus_oracle_m']['primary'])
        for who, first, T, idx, X, Xown, other_inner, other_name, other_pass in (('registry', l['twin_inner']['first_bag'], tb, ib, A, Bown, inB, rb, pb), ('revisit', l['rev_inner']['first_bag'], ta, ia, B, Aown, inA, ra, pa)):
            d, j = T.query(X[first][[0, 2]]); q = int(idx[j])
            r[who] = dict(first_bag=int(first), nearest_other=q, nearest_m=float(d), within_0_5=bool(d <= 0.5))
            if d <= 0.5: r[who].update(in_other_inner_rows=bool(q in other_inner), seen_from_other_lane_pass=bool(q in seen_by(other_name, other_pass[0], other_pass[1])), from_other_camera_line_m=abs(lat_of(other_name, other_pass[0], other_pass[1], Xown[q])))
        both = r['registry']['within_0_5'] and r['revisit']['within_0_5']
        r['kind'] = 'absent from the other census' if not both else ('present in both, outside one pass\'s inner rows' if not (r['registry']['in_other_inner_rows'] and r['revisit']['in_other_inner_rows']) else 'present in both and in both inner rows (station order differs)')
        rows.append(r)
    OUT[f'{ra}->{rb}'] = rows
    for r in rows: print(f"{ra}->{rb} lane {r['lane']} ({r['offset_m']:+.2f} m): {r['kind']}; registry {r['registry']}; revisit {r['revisit']}")
json.dump(dict(note='POST HOC, NOT A RULE', failures=OUT), open(sys.argv[1], 'w'), indent=1)
from collections import Counter; print(Counter(r['kind'] for v in OUT.values() for r in v))
