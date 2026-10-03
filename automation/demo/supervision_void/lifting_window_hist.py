#!/usr/bin/env python3
"""14 lifting_window_hist.py (nerf_new env): what the global-id lifting keeps from a SAM3 mask, as a picture (UJAMAA 2026-10-03;
Paul: "I don't understand the lifting blanking"). Per mask: the depths of the LiDAR returns inside it (its owned pixels), the
window the lifting keeps (from the nearest return to 1.5 m behind it, and no farther than 10 m from the camera), the count kept,
and the outcome (>= 25 kept: the mask takes the id of the 3D tree those points belong to; fewer: no id, supervision unlabelled)."""
from PIL import Image
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
import survey_paths
from pathlib import Path
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720
I = load_rig(str(S))['intrinsics']; proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S); m2g = json.load(open(G / 'global_ids.json'))['member_to_global']
CASES = [(3115, 1660, 'centre tree 146 at kf 3115'), (3112, 1617, 'centre tree 146 at kf 3112'), (3115, 1661, 'small tree 145 at kf 3115')]
fig, axs = plt.subplots(1, 3, figsize=(18, 4.6), dpi=100, sharey=False)
for ax, (K, oid, name) in zip(axs, CASES):
    uv, world, z = proj.project(K); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); cam = poses[K][:3, 3]
    ents = json.load(open(G / f'clip_{K // 200:03d}/frame_entries.json'))['frame_entries'][str(K % 200)]
    dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area'])) for e in ents]
    own = np.full((H, W), -1, np.int32)
    for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
    j = [i for i, d in enumerate(dec) if d[0] == oid][0]; sel = own[v, u] == j; zz = z[sel]; dist = np.linalg.norm(world[sel] - cam, axis=1)
    d0 = float(zz.min()); keep = (zz <= d0 + 1.5) & (dist <= 10.0); gid = m2g.get(f'{K // 200}:{oid}')
    ax.hist(zz, bins=np.arange(0, 20.5, 0.5), color='#9db8d9', label=f'{len(zz)} LiDAR returns inside the mask')
    ax.hist(zz[keep], bins=np.arange(0, 20.5, 0.5), color='#eb6834', label=f'{int(keep.sum())} kept by the rule')
    ax.axvspan(d0, d0 + 1.5, color='#eb6834', alpha=0.12); ax.axvline(d0, color='#eb6834', lw=1); ax.axvline(10, color='black', lw=1, ls='--')
    ax.text(d0 + 0.1, ax.get_ylim()[1] * 0.92, f'nearest return {d0:.1f} m:\nkeep up to {d0 + 1.5:.1f} m', fontsize=8, color='#b4441a', va='top')
    ax.text(10.1, ax.get_ylim()[1] * 0.55, '10 m limit', fontsize=8)
    ax.set_xlabel('depth from the camera (m)'); ax.set_ylabel('returns'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'{name}: ' + (f'kept {int(keep.sum())} >= 25 -> id {gid}' if gid is not None else f'kept {int(keep.sum())} < 25 -> no id, left unlabelled'), fontsize=10, loc='left')
fig.suptitle('The lifting rule: a SAM3 mask gets a tree id from the LiDAR returns inside it, keeping only those within 1.5 m of the NEAREST one (and within 10 m of the camera)', fontsize=10.5, x=0.01, ha='left')
fig.tight_layout(); fn = '/home/paperspace/data/demo_video_v2/citrus_a_bar/lifting_window_3112_3115.png'; fig.savefig(fn); print('[fig]', fn)
