#!/usr/bin/env python3
"""12 topdown_compare.py (nerf_new env): 01 hierarchy builds from above, one panel each (UJAMAA 2026-10-03; Paul: "run 01 on coral
with the strict solution ... redo"). Rows drawn as lines through their trees, classified as in rows_vs_lanes_multi.py: blue =
a tight line (RMS off-line <= 1 m) within 10 deg of the lanes, orange = a tight line more than 30 deg off (crosses the lanes),
grey dashed = scattered, not a line. Grey thin = rover path."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal')
_ts, P = survey_paths.poses_at_keyframes(S); p = P[:, [0, 2], 3]
st = np.diff(p, axis=0); n = np.linalg.norm(st, axis=1); k = n > 1e-3; u = st[k] / n[k, None]; w = n[k]
ang = (np.arctan2(u[:, 1], u[:, 0]) * 2) % (2 * np.pi); h, e = np.histogram(ang, 72, (0, 2 * np.pi), weights=w); hs = h + np.roll(h, 1) + np.roll(h, -1); pk = int(hs.argmax())
sel = np.zeros_like(ang, bool)
for d in (-1, 0, 1): sel |= (ang >= e[(pk + d) % 72]) & (ang < e[(pk + d) % 72 + 1])
la = np.arctan2((np.sin(ang[sel]) * w[sel]).sum(), (np.cos(ang[sel]) * w[sel]).sum()) / 2; L = np.array([np.cos(la), np.sin(la)])
BUILDS = [('CORAL, direction check 0.7: the prod hierarchy of 08-19 (marker_hierarchy_gen1.json)', S / 'prod/bateleur/scene_graph/marker_hierarchy_gen1.json'),
          ('CORAL, strict direction check 0.985, today\'s markers (experimental/coral_rows_20261003/strict_0985)', S / 'experimental/coral_rows_20261003/strict_0985/marker_hierarchy.json'),
          ('Census RANSAC solver, direction check 0.7: current prod (08-22)', S / 'prod/bateleur/scene_graph/marker_hierarchy.json')]
fig, axs = plt.subplots(len(BUILDS), 1, figsize=(18, 7.2 * len(BUILDS)), dpi=90)
for ax, (name, f) in zip(axs, BUILDS):
    H = json.load(open(f)); rows = {}
    for o in H['objects']:
        rows.setdefault(o['row_id'], []).append([o['xyz'][0], o['xyz'][2]])
    ax.plot(p[:, 0], p[:, 1], color='#d0d0d0', lw=0.7, zorder=1); cnt = {'lane': 0, 'diag': 0, 'scatter': 0, 'other': 0}
    for r, Q in rows.items():
        Q = np.array(Q)
        if r < 0: ax.scatter(Q[:, 0], Q[:, 1], s=30, marker='x', color='black', zorder=3); continue
        if len(Q) < 3: ax.scatter(Q[:, 0], Q[:, 1], s=14, color='#888888', zorder=3); continue
        c = Q - Q.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); a = float(np.degrees(np.arccos(min(1.0, abs(vt[0] @ L))))); rms = float(np.sqrt(np.mean((c @ vt[1]) ** 2)))
        kind = 'scatter' if rms > 1.0 else ('lane' if a <= 10 else ('diag' if a > 30 else 'other')); cnt[kind] += 1
        col = {'lane': '#2a78d6', 'diag': '#eb6834', 'scatter': '#777777', 'other': '#d9a400'}[kind]; Q = Q[np.argsort(Q @ vt[0])]
        ax.plot(Q[:, 0], Q[:, 1], '--' if kind == 'scatter' else '-', color=col, lw=1.8 if kind == 'diag' else 1.2, zorder=2); ax.scatter(Q[:, 0], Q[:, 1], s=13, color=col, zorder=3)
    ax.set_aspect('equal'); ax.set_xlim(-10, 225); ax.set_ylim(-3, 62)
    ax.set_title(f'{name}\n{len(H["objects"])} trees, {len([r for r in rows if r >= 0])} rows: {cnt["lane"]} follow the lanes (blue), {cnt["diag"]} straight lines across the lanes (orange), {cnt["scatter"]} scattered (grey dashed)', fontsize=10, loc='left')
fig.tight_layout(); fn = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar/rows_coral_strict_vs_prod_01.png'); fig.savefig(fn); print('[fig]', fn)
