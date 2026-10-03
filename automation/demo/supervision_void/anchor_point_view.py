#!/usr/bin/env python3
"""19 anchor_point_view.py (nerf_new env): the closest point of SAM3 #1617 on kf_003112 and the 1.5 m around it (UJAMAA 2026-10-03;
Paul: "show the central point, and the distance around it, I can't see how these lidar points are more than 1.5 metres away").
(a) Image, zoomed on the mask's left edge: every return inside the mask, coloured by depth, the closest one starred, a few depths
written next to their points. (b) From above in camera coordinates (x right, z = depth ahead): the camera, the mask's returns,
the closest point, the band the rule keeps (depth within 1.5 m of the closest point) and a 1.5 m circle around the closest point."""
from PIL import Image
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
from pathlib import Path
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720; K = 3112; OID = 1617
I = load_rig(str(S))['intrinsics']; fx, cx = float(I['fx']), float(I['cx'])
proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
ents = json.load(open(G / 'clip_015/frame_entries.json'))['frame_entries'][str(K % 200)]
dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area'])) for e in ents]
own = np.full((H, W), -1, np.int32)
for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
j = [i for i, d in enumerate(dec) if d[0] == OID][0]; m = own == j
uv, world, z = proj.project(K); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); ins = m[v, u]
ui, vi, zz = u[ins], v[ins], z[ins]; xc = (ui - cx) * zz / fx            # camera-frame x (m), right of the optical axis
a = int(np.argmin(zz)); za, xa = float(zz[a]), float(xc[a]); keep = zz <= za + 1.5
d3 = np.sqrt((xc - xa) ** 2 + (zz - za) ** 2 + (((vi - vi[a]) * zz - 0) / fx) ** 2 * 0)   # top-down distance from the closest point
tree = ~keep; dmin_tree = float(np.sqrt((xc[tree] - xa) ** 2 + (zz[tree] - za) ** 2).min())
print(f'closest point: depth {za:.2f} m at pixel ({ui[a]}, {vi[a]}); kept {int(keep.sum())} (depth <= {za + 1.5:.2f} m); '
      f'the other {int(tree.sum())} returns: depth {np.percentile(zz[tree], [0, 5, 50, 95]).round(2)}; nearest of them is {dmin_tree:.2f} m from the closest point (top-down)')
ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{K:06d}.png').convert('RGB'))
fig, axs = plt.subplots(1, 2, figsize=(20, 7.5), dpi=100, gridspec_kw=dict(width_ratios=[1.25, 1]))
x0, x1, y0, y1 = 340, 520, 200, 340; ax = axs[0]
ax.imshow(ph[y0:y1, x0:x1], extent=(x0, x1, y1, y0)); ax.contour(np.arange(x0, x1), np.arange(y0, y1), m[y0:y1, x0:x1], levels=[0.5], colors=['white'], linewidths=1.6, linestyles='--')
k = (ui >= x0) & (ui < x1) & (vi >= y0) & (vi < y1)
sc = ax.scatter(ui[k], vi[k], s=36, c=zz[k], cmap='turbo', vmin=2, vmax=10, edgecolors='black', linewidths=0.3)
ax.scatter(ui[keep], vi[keep], s=160, facecolors='none', edgecolors='white', linewidths=2)
ax.scatter([ui[a]], [vi[a]], s=420, marker='*', c='red', edgecolors='white', linewidths=1)
nb = np.nonzero(~keep & (np.abs(vi - vi[a]) <= 4) & (ui > ui[a]))[0]; q = nb[np.argmin(ui[nb])]
ax.annotate(f'closest point: {za:.1f} m deep\n(a tree 148 leaf)', (ui[a], vi[a]), xytext=(-70, 38), textcoords='offset points', fontsize=9.5, color='white',
            bbox=dict(boxstyle='round,pad=0.25', fc='#b71c1c', alpha=0.9, lw=0), arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
ax.annotate(f'nearest tree point on this line:\n{zz[q]:.1f} m deep, {zz[q] - za:.1f} m behind it', (ui[q], vi[q]), xytext=(30, -48), textcoords='offset points', fontsize=9.5, color='white',
            bbox=dict(boxstyle='round,pad=0.25', fc='#0d47a1', alpha=0.9, lw=0), arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off(); plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label='depth from the camera (m)')
ax.set_title(f'kf_003112, left edge of SAM3 #{OID}: red star = closest point ({za:.1f} m), white rings = the {int(keep.sum())} points kept.\nNeighbouring points on the same scan line are 7 m deep: next to it in the picture, ~4 m behind it in depth.', fontsize=10, loc='left')
ax = axs[1]
ax.axhspan(za, za + 1.5, color='#eb6834', alpha=0.15, label=f'kept: depth {za:.1f} to {za + 1.5:.1f} m (closest + 1.5 m)')
ax.add_patch(Circle((xa, za), 1.5, fill=False, ec='red', ls='--', lw=1.4, label='1.5 m around the closest point'))
ax.scatter(xc[tree], zz[tree], s=8, color='#2a78d6', label=f'{int(tree.sum())} points on the tree, thrown away')
ax.scatter(xc[keep], zz[keep], s=40, color='#eb6834', edgecolors='black', linewidths=0.4, label=f'{int(keep.sum())} points kept (tree 148 leaves)')
ax.scatter([xa], [za], s=300, marker='*', c='red', edgecolors='white', zorder=5)
ax.scatter([0], [0], s=120, marker='^', c='black', zorder=5); ax.text(0.15, 0.1, 'camera', fontsize=9)
ax.annotate('', xy=(xa, za + 1.5 + 0.05), xytext=(xa, za), arrowprops=dict(arrowstyle='<->', color='#b4441a', lw=1.5)); ax.text(xa + 0.15, za + 0.6, '1.5 m', color='#b4441a', fontsize=9)
ax.annotate('', xy=(xa, float(np.percentile(zz[tree], 5))), xytext=(xa, za), arrowprops=dict(arrowstyle='<->', color='black', lw=1.0, ls='--'))
ax.text(xa - 1.9, (za + float(np.percentile(zz[tree], 5))) / 2, f'{float(np.percentile(zz[tree], 5)) - za:.1f} m\nto the tree', fontsize=9)
ax.set_xlim(-4.5, 2.5); ax.set_ylim(-0.5, 10); ax.set_aspect('equal'); ax.set_xlabel('across the image, right of centre (m)'); ax.set_ylabel('depth ahead of the camera (m)')
ax.legend(fontsize=8, loc='upper left'); ax.set_title('From above (camera coordinates): the rule keeps only the orange band', fontsize=10, loc='left')
fig.tight_layout(); fn = '/home/paperspace/data/demo_video_v2/citrus_a_bar/kf003112_closest_point_window.png'; fig.savefig(fn); print('[fig]', fn)
