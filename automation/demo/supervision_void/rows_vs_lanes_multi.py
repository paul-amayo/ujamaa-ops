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
          ('census solver, dir 0.7, prod 08-22 10:42 (current)', S / 'prod/bateleur/scene_graph/marker_hierarchy.json'),
          ('CORAL, dir 0.7, current markers (10-03)', S / 'experimental/coral_rows_20261003/default_07/marker_hierarchy.json'),
          ('CORAL, dir 0.985 strict, current markers (10-03)', S / 'experimental/coral_rows_20261003/strict_0985/marker_hierarchy.json')]
OUT = {}
for name, f in BUILDS:
    if not f.exists(): print(f'{name}: missing {f}'); continue
    H = json.load(open(f)); objs = H['objects']; pv = H.get('_provenance', {})
    rows = {}
    for o in objs:
        if o['row_id'] >= 0: rows.setdefault(o['row_id'], []).append([o['xyz'][0], o['xyz'][2]])
    a, res, span = {}, {}, {}
    for r, Q in rows.items():
        Q = np.array(Q)
        if len(Q) >= 3:
            c = Q - Q.mean(0); _, sv, vt = np.linalg.svd(c, full_matrices=False); ax_ = vt[0]
            a[r] = float(np.degrees(np.arccos(min(1.0, abs(ax_ @ L))))); res[r] = float(np.sqrt(np.mean((c @ vt[1]) ** 2)))
            nrm = np.array([-L[1], L[0]]); span[r] = float(np.ptp(Q @ nrm))   # extent ACROSS the lanes
    line_diag = sorted(r for r in a if a[r] > 30 and res[r] <= 1.0)          # tight line (RMS off-line <= 1 m) crossing the lanes
    scatter = sorted(r for r in a if res[r] > 1.0)                           # not a line at all: catch-all / bucket
    lane_ok = sorted(r for r in a if a[r] <= 10 and res[r] <= 1.0)
    wide = sorted(r for r in a if span[r] > 1.5 * 6.4)                         # spans more than 1.5 lane spacings across the lanes
    print(f'{name}: {len(objs)} trees, {len(rows)} rows | lane-following lines {len(lane_ok)} | diagonal LINES across the lanes {len(line_diag)} '
          f'({sum(len(rows[r]) for r in line_diag)} trees) | scattered non-line rows {len(scatter)} ({sum(len(rows[r]) for r in scatter)} trees) | rows spanning > 1.5 lanes across {len(wide)}')
    OUT[name] = dict(file=str(f), diag=line_diag, scatter=scatter, wide=wide, angle={int(r): round(a[r]) for r in a}, rms={int(r): round(res[r], 2) for r in a}, span={int(r): round(span[r], 1) for r in a})
json.dump(OUT, open('/home/paperspace/data/citrus_all/01_13B_Jackal/experimental/coral_rows_20261003/rows_vs_lanes.json', 'w'), indent=1)
