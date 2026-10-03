#!/usr/bin/env python3
"""10 scene_graph_topdown.py (nerf_new env): the whole 01 scene graph from above (UJAMAA 2026-10-03; Paul: "show the top down
scene graph"). LIO ground plane (x, z); every tree a dot, each scene-graph row a line through its trees (ordered along the
row's PCA axis), coloured by its angle to the lane axis (the rover's length-weighted heading mode): blue = follows the lanes
(<= 10 deg), orange = cuts across (> 30 deg). Grey = rover path, black x = trees the fit left without a row, red = the camera
at kf_003112 (Citrus A)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal')
objs = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']
xz = {o['id']: np.array([o['xyz'][0], o['xyz'][2]]) for o in objs}; row_of = {o['id']: o['row_id'] for o in objs}
_ts, P = survey_paths.poses_at_keyframes(S); p = P[:, [0, 2], 3]
st = np.diff(p, axis=0); n = np.linalg.norm(st, axis=1); k = n > 1e-3; u = st[k] / n[k, None]; w = n[k]
ang = (np.arctan2(u[:, 1], u[:, 0]) * 2) % (2 * np.pi); h, e = np.histogram(ang, 72, (0, 2 * np.pi), weights=w); hs = h + np.roll(h, 1) + np.roll(h, -1); pk = int(hs.argmax())
sel = np.zeros_like(ang, bool)
for d in (-1, 0, 1): sel |= (ang >= e[(pk + d) % 72]) & (ang < e[(pk + d) % 72 + 1])
la = np.arctan2((np.sin(ang[sel]) * w[sel]).sum(), (np.cos(ang[sel]) * w[sel]).sum()) / 2; L = np.array([np.cos(la), np.sin(la)])
BLUE, ORANGE, GOLD = '#2a78d6', '#eb6834', '#d9a400'
fig, ax = plt.subplots(figsize=(18, 9), dpi=110)
ax.plot(p[:, 0], p[:, 1], color='#c8c8c8', lw=0.7, zorder=1)
nb = {'lane': 0, 'diag': 0, 'mid': 0}
for r in sorted({v for v in row_of.values() if v >= 0}):
    ids = [t for t in row_of if row_of[t] == r]; Q = np.array([xz[t] for t in ids])
    ax_ = np.linalg.svd(Q - Q.mean(0))[2][0] if len(Q) >= 2 else L; Q = Q[np.argsort(Q @ ax_)]
    a = float(np.degrees(np.arccos(min(1.0, abs(ax_ @ L))))); kind = 'lane' if a <= 10 else ('diag' if a > 30 else 'mid'); nb[kind] += 1
    c = {'lane': BLUE, 'diag': ORANGE, 'mid': GOLD}[kind]
    ax.plot(Q[:, 0], Q[:, 1], '-', color=c, lw=2.0 if kind == 'diag' else 1.3, alpha=0.9, zorder=2); ax.scatter(Q[:, 0], Q[:, 1], s=16, color=c, zorder=3)
    end = Q[np.argmax(Q[:, 1])]; ax.text(end[0], end[1] + 1.2, str(r), fontsize=7, color=c, ha='center', va='bottom', zorder=4)
for t in [t for t in row_of if row_of[t] < 0]: ax.scatter(*xz[t], s=30, marker='x', color='black', zorder=3)
c0 = p[3112]; hd = p[3122] - p[3102]; hd /= np.linalg.norm(hd)
ax.scatter(*c0, s=70, color='red', zorder=5); ax.annotate('', xy=c0 + 7 * hd, xytext=c0, arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=5)
ax.text(c0[0] + 1.5, c0[1] - 3.5, 'kf_003112\n(Citrus A)', fontsize=8, color='red', zorder=5)
for t in (146, 148): ax.text(xz[t][0] - 1.0, xz[t][1] + 0.8, str(t), fontsize=7, color='black', ha='right', zorder=5)
ax.set_aspect('equal'); ax.set_xlabel('LIO x (m)'); ax.set_ylabel('LIO z (m), the lane direction')
ax.set_title(f'01 scene graph from above: {len(xz)} trees, {sum(nb.values())} rows (labels = row id). Blue: {nb["lane"]} rows follow the lanes (within 10 deg). '
             f'Orange: {nb["diag"]} rows cut across them at 42-45 deg.\nGrey = rover path; black x = trees left without a row; red = camera at kf_003112, heading down the lane.', fontsize=10, loc='left')
fig.tight_layout(); fn = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar/scene_graph_topdown_01.png'); fig.savefig(fn); print('[fig]', fn, nb)
