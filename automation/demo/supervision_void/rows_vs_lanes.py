#!/usr/bin/env python3
"""6 rows_vs_lanes.py (nerf_new env): do the scene graph's rows follow the lanes the rover drove? (UJAMAA 2026-10-03)
In the LIO frame (ground plane x-z): the lane axis = length-weighted MODE of the rover's keyframe headings (the same rule as
build_marker_hierarchy._row_direction_from_trajectory); each scene-graph row's axis = PCA of its trees' (x, z). Reports the
angle of every row to the lane axis, which rows sit on BOTH sides of the lane at kf_003112, and writes a top-down figure
(trees coloured and labelled by row_id, the rover path, the camera at kf_003112)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal')
H = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))
print('hierarchy provenance: row_solver', H['_provenance'].get('row_solver'), '| dominant_direction_xz (row NORMAL)', H['_provenance'].get('dominant_direction_xz'), '| dir_threshold', H['_provenance'].get('dominant_direction_threshold'))
objs = H['objects']; xz = {o['id']: np.array([o['xyz'][0], o['xyz'][2]]) for o in objs}; row_of = {o['id']: o['row_id'] for o in objs}
_ts, P = survey_paths.poses_at_keyframes(S); p = P[:, [0, 2], 3]
st = np.diff(p, axis=0); n = np.linalg.norm(st, axis=1); k = n > 1e-3; u = st[k] / n[k, None]; w = n[k]
ang = (np.arctan2(u[:, 1], u[:, 0]) * 2) % (2 * np.pi); h, e = np.histogram(ang, 72, (0, 2 * np.pi), weights=w); hs = h + np.roll(h, 1) + np.roll(h, -1); pk = int(hs.argmax())
sel = np.zeros_like(ang, bool)
for d in (-1, 0, 1): sel |= (ang >= e[(pk + d) % 72]) & (ang < e[(pk + d) % 72 + 1])
lane = np.arctan2((np.sin(ang[sel]) * w[sel]).sum(), (np.cos(ang[sel]) * w[sel]).sum()) / 2; L = np.array([np.cos(lane), np.sin(lane)])
print(f'lane axis from the path (x,z) = ({L[0]:+.3f}, {L[1]:+.3f}); {100 * w[sel].sum() / w.sum():.0f}% of path length within +-7.5 deg of it')
def axis_deg(a, b): return float(np.degrees(np.arccos(min(1.0, abs(a @ b)))))
angs = {}
for r in sorted({v for v in row_of.values() if v >= 0}):
    Q = np.array([xz[t] for t in row_of if row_of[t] == r])
    if len(Q) >= 3: angs[r] = axis_deg(np.linalg.svd(Q - Q.mean(0))[2][0], L)
a = np.array(list(angs.values())); print(f'{len(a)} rows: angle to the lane axis median {np.median(a):.0f} deg, min {a.min():.0f}, max {a.max():.0f}; rows within 10 deg of the lanes: {int((a <= 10).sum())}')
print('rows 16-20:', {r: round(angs[r]) for r in range(16, 21) if r in angs})
i = 3112; c = p[i]; hd = p[i + 10] - p[i - 10]; hd /= np.linalg.norm(hd)   # poses are K-indexed (poses_at_keyframes returns (ts_ms, poses))
print(f'kf_003112: local travel direction vs lane axis {axis_deg(hd, L):.0f} deg')
side = lambda t: float(np.cross(hd, xz[t] - c)); ahead = lambda t: float(hd @ (xz[t] - c))
for t in (146, 147, 148, 156, 157):
    print(f'   tree {t} row {row_of[t]}: {ahead(t):+.1f} m ahead, {side(t):+.1f} m {"left" if side(t) > 0 else "right"} of the camera path (sign by cross product)')
diag = sorted(r for r, d in angs.items() if d > 30); nd = sum(1 for t in row_of if row_of[t] in diag)
print(f'rows more than 30 deg off the lanes: {diag} ({nd} of {len(row_of)} trees)')
fig, ax = plt.subplots(figsize=(10, 10), dpi=100); cm = plt.get_cmap('tab10'); R0 = 22
ax.plot(p[:, 0], p[:, 1], color='#bbbbbb', lw=0.8, zorder=1)
for r in angs:
    ids = [t for t in row_of if row_of[t] == r]; Q = np.array([xz[t] for t in ids]); ax_ = np.linalg.svd(Q - Q.mean(0))[2][0]; Q = Q[np.argsort(Q @ ax_)]
    hot = r in (16, 17, 18, 19, 20); col = cm(r % 10)
    ax.plot(Q[:, 0], Q[:, 1], '-', color=col, lw=2.6 if hot else 0.9, alpha=0.95 if hot else 0.45, zorder=2)
    ax.scatter(Q[:, 0], Q[:, 1], s=34, color=col, zorder=3)
    j = int(np.argmin(np.linalg.norm(Q - c, axis=1))); ax.text(Q[j, 0] + 0.5, Q[j, 1] - 1.2, f'row {r}', fontsize=8 if hot else 6, weight='bold' if hot else 'normal', color=col, clip_on=True, zorder=4)
ax.annotate('', xy=c + 5 * hd, xytext=c, arrowprops=dict(arrowstyle='->', color='red', lw=2.2)); ax.scatter(*c, s=90, color='red', zorder=5)
for t in (146, 147, 148, 156, 157): ax.text(xz[t][0] - 2.2, xz[t][1] + 0.6, f'{t}', fontsize=8, color='red', clip_on=True, zorder=5)
ax.set_xlim(c[0] - R0, c[0] + R0); ax.set_ylim(c[1] - R0, c[1] + R0); ax.set_aspect('equal'); ax.set_xlabel('LIO x (m)'); ax.set_ylabel('LIO z (m)')
ax.set_title(f'01 scene graph, top-down: trees joined by row_id (rows 16-20 bold); grey = rover path; red = camera at kf_003112.\n'
             f'The rover drives the lanes (vertical). Rows 16-20 run at 42-45 deg across them; {len(angs) - len(diag)} of {len(angs)} rows follow the lanes.', fontsize=9, loc='left')
fn = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar/rows_vs_lanes_01_kf003112.png'); fig.savefig(fn, bbox_inches='tight'); print('[fig]', fn)
