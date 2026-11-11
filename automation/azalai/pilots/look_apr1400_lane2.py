#!/usr/bin/env python3
"""look_apr1400_lane2.py (A100 agent, 11 Nov 2026): AZ2 post hoc look (no verdict): why the first inner-row bag of the causal
list (graph poses) of April 15 14:00 lane 2 lies 2.6 m before the census's row end. Lists the census bags near the row start
with their station and lateral offset (whole-lane camera line), their lifted returns from the lane pass and from before it,
and whether they are inner-row bags."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azalai_online_geom import load_pass, dstation
CAUSAL = Path('/mnt/shared/azalai_online/causal')
Rv = load_pass('apr1400'); lr = Rv['lanes'][1]; c2w = Rv['c2w']; B = Rv['B']; k0, k1, s = lr['k0'], lr['k1'], lr['sense']
P = c2w[k0:k1 + 1, :3, 3][:, [0, 2]]; m = P.mean(0); w, V = np.linalg.eigh(np.cov((P - m).T)); d = V[:, np.argmax(w)]; d = d if d[1] > 0 else -d; nrm = np.array([d[1], -d[0]])
z = np.load(CAUSAL / 'causal_apr1400_graph.npz', allow_pickle=True); W = z['W']; sel = np.flatnonzero(z['step_lane'] == 1); Pw = W[k0:k1 + 1, :3, 3]
for j in range(k1 - k0 + 1):
    if np.linalg.norm(Pw[j] - Pw[0]) < 1.0: continue
    Q = Pw[: j + 1]; q = Q.mean(0); ww, VV = np.linalg.eigh((Q - q).T @ (Q - q)); ax = VV[:, np.argmax(ww)]; ax = ax if ax @ (Pw[j] - Pw[0]) > 0 else -ax
    up = -W[k0:k0 + j + 1, :3, 1].mean(0); up -= (up @ ax) * ax; up /= np.linalg.norm(up); right = np.cross(ax, up)
    st = sel[j]; Xb = z['bags_xyz'][int(z['step_start'][st]): int(z['step_start'][st]) + int(z['step_n'][st])]; inn = np.abs((Xb - q) @ right) <= 1.5
    if inn.any():
        xb = Xb[inn][np.argmin((Xb[inn] - Pw[0]) @ ax)]
        print(f'first belief at keyframe {k0 + j}: first causal inner-row bag at station {dstation(lr, xb):+.2f} m, lateral {float((xb[[0, 2]] - m) @ nrm):+.2f} m from the whole-lane camera line, {float(abs((xb - q) @ right)):.2f} m from the causal line')
        break
print(f'lane 2: keyframes {k0}..{k1}, sense {s}')
fin = np.isfinite(B).all(1); ids = np.flatnonzero(fin); st_ = np.array([dstation(lr, B[b]) for b in ids]); lat = (B[ids][:, [0, 2]] - m) @ nrm
near = (st_ > -5) & (st_ < 1.5) & (np.abs(lat) < 2.6); inner = set(lr['ids'].tolist())
for b, sb, lb in sorted(zip(ids[near], st_[near], lat[near]), key=lambda t: t[1]):
    selb = Rv['ob'] == b; ks = Rv['ok'][selb]; ns = Rv['on'][selb]
    print(f'census bag {b:4d} station {sb:+6.2f} lateral {lb:+5.2f} returns lane pass {int(ns[(ks >= k0) & (ks <= k1)].sum()):5d} before it {int(ns[ks < k0].sum()):5d} inner row {"yes" if b in inner else "no"}')
