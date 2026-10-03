#!/usr/bin/env python3
"""9 frame_lidar_sam3.py (nerf_new env): one keyframe, two panels (UJAMAA 2026-10-03; Paul: "show all lidar points projected
to frame 3112, and then sam3 masks on this same image"). Top: every laser return the lifting's LaserProjector puts in the frame
(rig intrinsics, Jackal default l2c, 80 ms match), coloured by camera depth. Bottom: every SAM3 detection on the frame, filled
by the painter's smallest-on-top partition, outlined in full, labelled with its global id (and scene-graph row) or "no id"
(dashed red). Usage: frame_lidar_sam3.py [K] (default 3112)."""
from PIL import Image                     # PIL BEFORE aru_nerf_interface
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3112
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720
rig = load_rig(str(S)); I = rig['intrinsics']
proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
uv, world, z = proj.project(K); u, v = uv[:, 0], uv[:, 1]
m2g = json.load(open(G / 'global_ids.json'))['member_to_global']
row_of = {o['id']: o['row_id'] for o in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
ents = json.load(open(G / f'clip_{K // 200:03d}/frame_entries.json'))['frame_entries'].get(str(K % 200), [])
dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area']), float(e['score'])) for e in ents]
own = np.full((H, W), -1, np.int32)
for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{K:06d}.png').convert('RGB'))
gids = {j: m2g.get(f'{K // 200}:{oid}') for j, (oid, m, area, sc_) in enumerate(dec)}
PAL = [(42, 120, 214), (27, 175, 122), (237, 161, 0), (232, 123, 164), (74, 58, 167), (0, 160, 190), (150, 110, 60), (120, 190, 40)]   # no red: red = no id
gcol = {g: np.array(PAL[i % len(PAL)], np.float32) for i, g in enumerate(sorted({g for g in gids.values() if g is not None}))}
near = (z < 4.5); jn = own[v.astype(int), u.astype(int)]
j17 = [j for j, d in enumerate(dec) if gids[j] is None and d[2] > 20000]; j48 = [j for j, d in enumerate(dec) if gids[j] == 148]
if j17 and j48:
    sel = near & (jn == j17[0]); in148 = dec[j48[0]][1][v[sel].astype(int), u[sel].astype(int)]
    print(f'near (<4.5 m) returns owned by the no-id centre mask {dec[j17[0]][0]}: {int(sel.sum())}; also inside 148\'s own SAM3 mask: {int(in148.sum())}; pixels {list(zip(u[sel].astype(int), v[sel].astype(int), np.round(z[sel], 2)))}')
fig, axs = plt.subplots(3, 1, figsize=(12.8, 22.4), dpi=100, gridspec_kw=dict(height_ratios=[1, 1, 1]))
ax = axs[0]; ax.imshow(ph); sc = ax.scatter(u, v, s=4, c=z, cmap='turbo', vmin=1, vmax=20, linewidths=0)
ax.set_title(f'kf_{K:06d}: all {len(z)} LiDAR returns projected into the frame, coloured by depth (m)', fontsize=11, loc='left'); ax.set_axis_off()
plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
ax = axs[1]; base = ph.astype(np.float32) * 0.55
for j in range(len(dec)):
    if gids[j] is not None: base[own == j] = base[own == j] * 0.35 + gcol[gids[j]] * 0.65
ax.imshow(base.astype(np.uint8)); lab = []
for j, (oid, m, area, s_) in enumerate(dec):
    g = gids[j]; o = own == j
    ax.contour(m, levels=[0.5], colors=['red' if g is None else [c / 255 for c in gcol[g]]], linewidths=1.6, linestyles='--' if g is None else '-')
    if o.sum() >= 1000:
        yy, xx = np.nonzero(o); lab.append((float(np.median(xx)), float(np.median(yy)), (f'tree {g} (row {row_of.get(g)})' if g is not None else 'no id') + f'\n{o.sum() / 1000:.1f}k px', g))
lab.sort(); xs = np.linspace(70, W - 70, len(lab))
for i, (x0, y0, t, g) in enumerate(lab):
    ax.annotate(t, xy=(x0, y0), xytext=(xs[i], 60 if i % 2 == 0 else 150), fontsize=8, color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='red' if g is None else tuple(gcol[g] / 255), alpha=0.9, lw=0), arrowprops=dict(arrowstyle='-', color='white', lw=0.8))
ax.set_title(f'kf_{K:06d}: all {len(dec)} SAM3 detections. Filled = pixels painted with a tree id (colour per tree); dashed red outline = no id, painted unlabelled', fontsize=11, loc='left'); ax.set_axis_off()
ax = axs[2]; x0, x1, y0, y1 = 300, 620, 170, 400
ax.imshow(ph[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
for j, (oid, m, area, s_) in enumerate(dec):
    g = gids[j]
    if m[y0:y1, x0:x1].any(): ax.contour(np.arange(x0, x1), np.arange(y0, y1), m[y0:y1, x0:x1], levels=[0.5], colors=['red' if g is None else [c / 255 for c in gcol[g]]], linewidths=2.2, linestyles='--' if g is None else '-')
k = (u >= x0) & (u < x1) & (v >= y0) & (v < y1); sc = ax.scatter(u[k], v[k], s=22, c=z[k], cmap='turbo', vmin=1, vmax=20, edgecolors='white', linewidths=0.3)
if j17: sel = near & (jn == j17[0]); ax.scatter(u[sel], v[sel], s=160, facecolors='none', edgecolors='white', linewidths=2.0)
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off(); plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
ax.set_title('Zoom on the centre tree\'s left edge: SAM3 outlines (blue = tree 148, dashed red = the centre tree, no id) and LiDAR returns by depth; white rings = the near (<4.5 m) returns inside the centre-tree mask that set its depth window', fontsize=9.5, loc='left', wrap=True)
fig.tight_layout(); fn = Path(f'/home/paperspace/data/demo_video_v2/citrus_a_bar/kf{K:06d}_lidar_and_sam3.png'); fig.savefig(fn); print('[fig]', fn)
