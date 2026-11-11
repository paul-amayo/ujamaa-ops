#!/usr/bin/env python3
"""azalai_online_odo.py (A100 agent, 11 Nov 2026): AZ2 online pilot, post hoc (not a rule, no verdict): how far the plain
laser odometry's travel along each lane pass differs from the graph's, per revisit pass and lane. For keyframe k of a lane
pass k0..k1: travel along the lane's own principal direction (fitted separately to each source's camera centres) from k0,
laser odometry minus graph; reported at the lane's end and as the largest absolute value over the lane. It says what part of
the station error the odometry contributes (the rest is the datum).
  azalai_online_odo.py --causal <dir> --passes apr1130,apr1400,apr16_1230,apr1054 --out <json>"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azalai_online_geom import lane_passes, EXP

ap = argparse.ArgumentParser(); ap.add_argument('--causal', required=True); ap.add_argument('--passes', required=True); ap.add_argument('--out', required=True); A = ap.parse_args()
out = {}
for name in A.passes.split(','):
    c2w = np.array(EXP[name]['c2w'], float); z = np.load(Path(A.causal) / f'causal_{name}_lo.npz', allow_pickle=True); W = z['W']; rows = []
    for i, (k0, k1, s) in enumerate(lane_passes(c2w)):
        if not bool(z['lane_ok'][i]): rows.append(dict(lane=i + 1, lo=False)); continue
        def travel(P):
            q = P.mean(0); w, V = np.linalg.eigh((P - q).T @ (P - q)); ax = V[:, np.argmax(w)]; ax = ax if ax @ (P[-1] - P[0]) > 0 else -ax; return (P - P[0]) @ ax
        tg = travel(c2w[k0:k1 + 1, :3, 3]); tl = travel(W[k0:k1 + 1, :3, 3]); d = tl - tg
        rows.append(dict(lane=i + 1, lo=True, length_m=float(tg[-1]), diff_end_m=float(d[-1]), diff_absmax_m=float(np.abs(d).max()), at_m=float(tg[int(np.argmax(np.abs(d)))])))
    out[name] = rows
    ok = [r for r in rows if r['lo']]
    print(f"{name}: lanes with laser odometry {len(ok)}; LO minus graph travel at the lane end (m): " + ', '.join(f"{r['lane']}:{r['diff_end_m']:+.2f}" for r in ok)
          + f"; largest |difference| over a lane: {max(r['diff_absmax_m'] for r in ok):.2f} m (lane {max(ok, key=lambda r: r['diff_absmax_m'])['lane']})")
json.dump(out, open(A.out, 'w'), indent=1)
