#!/usr/bin/env python3
"""11 rows_vs_lanes_multi.py (nerf_new env): row alignment to the lanes for several 01 hierarchy builds (UJAMAA 2026-10-03; Paul:
"coral would never make rows like this, where do they come from?"). Lane axis = the rover's length-weighted heading mode (as in
rows_vs_lanes.py); per build: rows, rows within 10 deg of the lanes, rows more than 30 deg off, and the trees in those."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal')
_ts, P = survey_paths.poses_at_keyframes(S); p = P[:, [0, 2], 3]
st = np.diff(p, axis=0); n = np.linalg.norm(st, axis=1); k = n > 1e-3; u = st[k] / n[k, None]; w = n[k]
ang = (np.arctan2(u[:, 1], u[:, 0]) * 2) % (2 * np.pi); h, e = np.histogram(ang, 72, (0, 2 * np.pi), weights=w); hs = h + np.roll(h, 1) + np.roll(h, -1); pk = int(hs.argmax())
sel = np.zeros_like(ang, bool)
for d in (-1, 0, 1): sel |= (ang >= e[(pk + d) % 72]) & (ang < e[(pk + d) % 72 + 1])
la = np.arctan2((np.sin(ang[sel]) * w[sel]).sum(), (np.cos(ang[sel]) * w[sel]).sum()) / 2; L = np.array([np.cos(la), np.sin(la)])
BUILDS = [('CORAL, prod 08-19 (marker_hierarchy_gen1.json)', S / 'prod/bateleur/scene_graph/marker_hierarchy_gen1.json'),
          ('census solver, dir 0.7, pilot 08-22 07:07', S / 'experimental/gen2_m0/01/marker_hierarchy.json'),
          ('census solver, dir 0.985, pilot 08-22 07:26', S / 'experimental/gen2_m0/01_dir985/marker_hierarchy.json'),
          ('census solver, dir 0.7, prod 08-22 10:42 (current)', S / 'prod/bateleur/scene_graph/marker_hierarchy.json')]
for name, f in BUILDS:
    if not f.exists(): print(f'{name}: missing {f}'); continue
    H = json.load(open(f)); objs = H['objects']; pv = H.get('_provenance', {})
    rows = {}
    for o in objs:
        if o['row_id'] >= 0: rows.setdefault(o['row_id'], []).append([o['xyz'][0], o['xyz'][2]])
    a = {}
    for r, Q in rows.items():
        Q = np.array(Q)
        if len(Q) >= 3: a[r] = float(np.degrees(np.arccos(min(1.0, abs(np.linalg.svd(Q - Q.mean(0))[2][0] @ L)))))
    diag = sorted(r for r, d in a.items() if d > 30)
    print(f'{name}: {len(objs)} trees, {len(rows)} rows ({len(a)} with >= 3 trees); within 10 deg of the lanes {sum(1 for d in a.values() if d <= 10)}, '
          f'10-30 deg {sum(1 for d in a.values() if 10 < d <= 30)}, more than 30 deg {len(diag)} holding {sum(len(rows[r]) for r in diag)} trees'
          f' | census gids {pv.get("census", {}).get("n_census", "-")}, models made {pv.get("census", {}).get("n_models_made", "-")}')
