#!/usr/bin/env python3
"""13 rows_vs_lanes_survey.py (nerf_new env): do a survey's scene-graph rows follow the lanes its rover drove? Any survey root.
(UJAMAA 2026-10-03; Paul: "see if this census error bled into 04 and klapmuts".) Lane axis = the rover's length-weighted heading
mode (build_marker_hierarchy's trajectory rule). A row with >= 3 trees is: along the lanes (<= 10 deg), a straight DIAGONAL
(> 30 deg off and RMS off-line <= 0.25 x the survey's row spacing), or scattered (RMS above that). Row spacing = median gap
between adjacent lane-following rows across the lanes. Usage: rows_vs_lanes_survey.py <root> [<root> ...] [--hier <json>]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
args = sys.argv[1:]; hier = None
if '--hier' in args: i = args.index('--hier'); hier = args[i + 1]; args = args[:i] + args[i + 2:]
for root in args:
    S = Path(root); f = Path(hier) if hier else S / 'prod/bateleur/scene_graph/marker_hierarchy.json'
    H = json.load(open(f)); pv = H.get('_provenance', {})
    try:
        _ts, P = survey_paths.poses_at_keyframes(S); p = P[:, [0, 2], 3]; p = p[~np.isnan(p).any(1)]
    except Exception as ex:
        print(f'{S.name}: no keyframe poses ({ex})'); continue
    st = np.diff(p, axis=0); n = np.linalg.norm(st, axis=1); k = n > 1e-3; u = st[k] / n[k, None]; w = n[k]
    ang = (np.arctan2(u[:, 1], u[:, 0]) * 2) % (2 * np.pi); h, e = np.histogram(ang, 72, (0, 2 * np.pi), weights=w); hs = h + np.roll(h, 1) + np.roll(h, -1); pk = int(hs.argmax())
    sel = np.zeros_like(ang, bool)
    for d in (-1, 0, 1): sel |= (ang >= e[(pk + d) % 72]) & (ang < e[(pk + d) % 72 + 1])
    la = np.arctan2((np.sin(ang[sel]) * w[sel]).sum(), (np.cos(ang[sel]) * w[sel]).sum()) / 2; L = np.array([np.cos(la), np.sin(la)]); nrm = np.array([-L[1], L[0]])
    rows = {}
    for o in H['objects']: rows.setdefault(o['row_id'], []).append([o['xyz'][0], o['xyz'][2]])
    st_ = {}
    for r, Q in rows.items():
        Q = np.array(Q)
        if r < 0 or len(Q) < 3: continue
        c = Q - Q.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False)
        st_[r] = (float(np.degrees(np.arccos(min(1.0, abs(vt[0] @ L))))), float(np.sqrt(np.mean((c @ vt[1]) ** 2))), float(Q.mean(0) @ nrm), len(Q))
    lane = sorted(v[2] for v in st_.values() if v[0] <= 10); sp = float(np.median(np.diff(lane))) if len(lane) > 2 else float('nan')
    tight = 0.25 * sp if sp == sp else 1.0
    diag = [r for r, v in st_.items() if v[0] > 30 and v[1] <= tight]; scat = [r for r, v in st_.items() if v[1] > tight]; along = [r for r, v in st_.items() if v[0] <= 10 and v[1] <= tight]
    print(f'{S.name}: {pv.get("row_solver", "pre-census")} dir {pv.get("dominant_direction_threshold")} | {len(H["objects"])} objects, {len([r for r in rows if r >= 0])} rows, '
          f'{sum(len(rows[r]) for r in rows if r < 0)} row-less | lane axis holds {100 * w[sel].sum() / w.sum():.0f}% of path, row spacing {sp:.2f} m | '
          f'along the lanes {len(along)} | straight diagonals {len(diag)} ({sum(st_[r][3] for r in diag)} objects) | scattered {len(scat)} ({sum(st_[r][3] for r in scat)} objects)'
          + (f' | diagonal rows {sorted(diag)} at {sorted(round(st_[r][0]) for r in diag)} deg' if diag else ''))
