#!/usr/bin/env python3
"""azalai_online_datum.py (A100 agent, 11 Nov 2026): AZ2 online pilot, the datum diagnostic declared in
design/AZ2_short_gap_note.md (585175f), computed after scoring and carrying no verdict.

For every read pair and lane it splits the station error at the first belief into two parts measured without the robot:
  (i)  the revisit census's row-end bag (its extreme inner-row bag at the entry end) carried into the registry's lane frame
       through the site-wide fits, i.e. where the revisit's row starts on the registry's stations (0 = same bag);
  (ii) the first inner-row bag of the causal list at the first belief, against the revisit census's row-end bag, along the
       revisit's own lane frame (graph-pose list for b; for M the laser-odometry list mapped to the graph by the robot's own
       pose at that keyframe), i.e. how far into the row the first bag the lane pass itself sees lies.
Then e_first is about -((i) + (ii)) plus the odometry and axis error of the first metres. Also listed: how many of the
revisit census's inner-row bags lie before the first causal bag, and what share of the census row-end bag's lifted
returns come from the lane pass's own keyframes (the rest come from the turn-in before it).
Lane passes, inner rows and lane frames are rebuilt as in azalai_online_resim.py (same author; from the design text).
  azalai_online_datum.py --pairs dec:apr1130,dec:apr1400,dec:apr16_1230,apr1130:apr1400,apr1130:apr16_1230 --records <dir> --causal <dir> --out <file>"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for k in ('pairs', 'records', 'causal', 'out'): ap.add_argument('--' + k, required=True)
A = ap.parse_args()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azalai_online_geom import load_pass, dstation, placed


def first_bag(rev, src, lane, c2w):
    """the causal list's first inner-row bag at the first keyframe with a belief, in the revisit's graph world"""
    z = np.load(Path(A.causal) / f'causal_{rev}_{src}.npz', allow_pickle=True); i = lane['i']
    if not bool(z['lane_ok'][i]): return None
    k0, k1, s = lane['k0'], lane['k1'], lane['sense']; W = z['W']; P = W[k0:k1 + 1, :3, 3]; sel = np.flatnonzero(z['step_lane'] == i)
    for j in range(k1 - k0 + 1):
        if np.linalg.norm(P[j] - P[0]) < 1.0: continue
        Q = P[: j + 1]; q = Q.mean(0); w, V = np.linalg.eigh((Q - q).T @ (Q - q)); ax = V[:, np.argmax(w)]; ax = ax if ax @ (P[j] - P[0]) > 0 else -ax
        up = -W[k0:k0 + j + 1, :3, 1].mean(0); up = up - (up @ ax) * ax; up /= np.linalg.norm(up); right = np.cross(ax, up)
        st = sel[j]; Xb = z['bags_xyz'][int(z['step_start'][st]): int(z['step_start'][st]) + int(z['step_n'][st])]
        inn = np.abs((Xb - q) @ right) <= 1.5
        if not inn.any(): continue
        xb = Xb[inn][np.argmin((Xb[inn] - P[0]) @ ax)]
        T = c2w[k0 + j] @ np.linalg.inv(W[k0 + j])          # the source's world -> graph world, through the robot's pose at this keyframe
        return dict(j=j, k=k0 + j, x=(T[:3, :3] @ xb + T[:3, 3]))
    return None


rows = []
for pr in A.pairs.split(','):
    reg, rev = pr.split(':'); tag = f'{reg}_{rev}'; Rg, Rv = load_pass(reg), load_pass(rev); G = np.linalg.inv(placed(reg)) @ placed(rev)
    S = json.load(open(Path(A.records) / f'online_summary_{tag}.json')); half = S['half_spacing_m']; lanes_s = {l['lane']: l for l in S['lanes']}
    for i, (lt, lr) in enumerate(zip(Rg['lanes'], Rv['lanes'])):
        lr['i'] = i
        if lt['frame'] is None or lr['frame'] is None: continue
        r_end = int(lr['ids'][np.argmin(lr['sig'])]); t_end = int(lt['ids'][np.argmin(lt['sig'])])
        part_i = dstation(lt, G[:3, :3] @ Rv['B'][r_end] + G[:3, 3])
        sel = Rv['ob'] == r_end; ks, ns = Rv['ok'][sel], Rv['on'][sel]; share_lane = float(ns[(ks >= lr['k0']) & (ks <= lr['k1'])].sum() / max(ns.sum(), 1))
        e = dict(pair=tag, lane=i + 1, sense=lr['sense'], half=half, registry_row_end_bag=t_end, revisit_row_end_bag=r_end, part_i_m=part_i,
                 row_end_bag_lane_share=share_lane, row_end_bag_returns=int(ns.sum()))
        for v, src in (('b', 'graph'), ('M', 'lo')):
            fb = first_bag(rev, src, lr, Rv['c2w']); ef = lanes_s[i + 1]['var'][v]['err_first_m'] if lanes_s[i + 1].get('var', {}).get(v) else None
            if fb is None: e[v] = None; continue
            part_ii = dstation(lr, fb['x']); n_before = int((lr['sig'] < part_ii - 0.3).sum())
            e[v] = dict(err_first_m=ef, first_belief_k=fb['k'], part_ii_m=part_ii, census_bags_before_first_causal=n_before, predicted_err_first_m=-(part_i + part_ii))
        rows.append(e)
json.dump(rows, open(A.out, 'w'), indent=1)
print(f'{"pair":20s} lane  half   (i)    | b: e_first (ii)  pred  n_before | M: e_first (ii)  pred  n_before | row-end bag: lane share of returns')
for e in rows:
    fmt = lambda v: '   n/a' if e[v] is None else f"{e[v]['err_first_m']:+6.2f} {e[v]['part_ii_m']:+5.2f} {e[v]['predicted_err_first_m']:+5.2f} {e[v]['census_bags_before_first_causal']:3d}"
    flag = '*' if (e['M'] and abs(e['M']['err_first_m']) > e['half']) else ' '
    print(f"{e['pair']:20s} {e['lane']:3d}{flag} {e['half']:.2f} {e['part_i_m']:+6.2f} | {fmt('b')}      | {fmt('M')}      | {e['row_end_bag_lane_share']:.2f} of {e['row_end_bag_returns']}")
