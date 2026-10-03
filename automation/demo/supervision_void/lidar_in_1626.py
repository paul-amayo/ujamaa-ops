#!/usr/bin/env python3
"""17 lidar_in_1626.py (nerf_new env): the LiDAR returns that land in SAM3 #1626 on kf_003112 (UJAMAA 2026-10-03; Paul: "show me
lidar points that land in 1626"). (a) On the image: the returns inside the raw mask, coloured by depth, over the mask outline.
(b) From above (LIO x-z): the same returns, coloured by the 3D tree box they fall in, with the boxes, the camera and its heading."""
from PIL import Image
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
import survey_paths
from pathlib import Path
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720; K = 3112; OID = 1626
I = load_rig(str(S))['intrinsics']; proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S); st = {int(k): v for k, v in json.load(open(G / 'global_ids.json'))['stats'].items()}
ents = json.load(open(G / 'clip_015/frame_entries.json'))['frame_entries'][str(K % 200)]
m = [mu.decode(e['rle']).astype(bool) for e in ents if int(e['obj_id']) == OID][0]
uv, world, z = proj.project(K); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); ins = m[v, u]
P, zz, ui, vi = world[ins], z[ins], u[ins], v[ins]
def box(g): s = st[g]; return np.all((P >= np.array(s['world_bbox_min'])) & (P <= np.array(s['world_bbox_max'])), axis=1)
B = {146: box(146), 145: box(145), 148: box(148)}; none = ~(B[146] | B[145] | B[148])
COL = {146: '#eb6834', 145: '#2a78d6', 148: '#555555'}
ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{K:06d}.png').convert('RGB'))
fig, axs = plt.subplots(1, 2, figsize=(20, 7), dpi=100, gridspec_kw=dict(width_ratios=[1.6, 1]))
x0, x1, y0, y1 = 330, 700, 160, 400; ax = axs[0]
ax.imshow(ph[y0:y1, x0:x1], extent=(x0, x1, y1, y0)); ax.contour(np.arange(x0, x1), np.arange(y0, y1), m[y0:y1, x0:x1], levels=[0.5], colors=['white'], linewidths=1.6, linestyles='--')
sc = ax.scatter(ui, vi, s=16, c=zz, cmap='turbo', vmin=2, vmax=16, edgecolors='black', linewidths=0.2); plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label='depth (m)')
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off()
ax.set_title(f'kf_003112: the {int(ins.sum())} LiDAR returns inside SAM3 #{OID} (dashed white), coloured by depth', fontsize=10.5, loc='left')
ax = axs[1]; cam = poses[K][:3, 3]; fwd = poses[K + 10][:3, 3] - poses[K - 10][:3, 3]; fwd /= np.linalg.norm(fwd)
for g, c in COL.items():
    s = st[g]; lo, hi = np.array(s['world_bbox_min']), np.array(s['world_bbox_max'])
    ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec=c, lw=1.4, ls='--'))
    ax.text(hi[0] + 0.2, hi[2] - 0.6, f'tree {g} box', color=c, fontsize=9)
    ax.scatter(P[B[g], 0], P[B[g], 2], s=10, color=c, label=f'in tree {g} box ({int(B[g].sum())})')
ax.scatter(P[none, 0], P[none, 2], s=10, color='#bbbbbb', label=f'in none of these ({int(none.sum())})')
ax.scatter(*cam[[0, 2]], s=80, color='red', zorder=5); ax.annotate('', xy=cam[[0, 2]] + 4 * fwd[[0, 2]] / np.linalg.norm(fwd[[0, 2]]), xytext=cam[[0, 2]], arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(cam[0] + 0.4, cam[2] - 0.8, 'camera', color='red', fontsize=9)
dc = np.linalg.norm(P - cam, axis=1); lo6, hi6 = np.array(st[146]['world_bbox_min']), np.array(st[146]['world_bbox_max'])
near146 = none & np.all((P >= lo6 - 1.0) & (P <= hi6 + 1.0), axis=1); mid = none & ~near146 & (dc < 30); far = none & (dc >= 30)
print(f'unboxed {int(none.sum())}: within 1 m of tree 146 box {int(near146.sum())} (depth {np.percentile(zz[near146], [5, 50, 95]).round(1) if near146.any() else "-"}), '
      f'other < 30 m {int(mid.sum())} (distance {np.percentile(dc[mid], [5, 50, 95]).round(1) if mid.any() else "-"}), beyond 30 m {int(far.sum())}')
ax.set_xlim(cam[0] - 7, cam[0] + 7); ax.set_ylim(cam[2] - 2, cam[2] + 26); ax.set_aspect('equal'); ax.set_xlabel('LIO x (m)'); ax.set_ylabel('LIO z (m), down the lane')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0)); ax.text(cam[0] - 6.5, cam[2] + 25, f'{int(far.sum())} more returns lie beyond 30 m (seen through gaps)', fontsize=8, va='top')
ax.set_title('The same returns from above, coloured by the 3D tree box they fall in', fontsize=10.5, loc='left')
fig.tight_layout(); fn = '/home/paperspace/data/demo_video_v2/citrus_a_bar/kf003112_lidar_in_1626.png'; fig.savefig(fn); print('[fig]', fn, {g: int(b.sum()) for g, b in B.items()}, int(none.sum()))
