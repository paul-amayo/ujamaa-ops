#!/usr/bin/env python3
"""5 lidar_in_mask.py (nerf_new env): the RAW laser returns inside a SAM3 mask on its keyframe, before the lifting's gates
(UJAMAA 2026-10-03; Paul: "surely we should get lidar returns on the closest tree?"). Builds the LaserProjector exactly as
cluster_tree_instances.main() does (rig intrinsics + l2c, kf20cm image mono, transform_lio, laser mono, 80 ms match), takes the
detection's OWNED pixels under collect()'s partition hygiene (smallest-on-top, no largest-component step), and reports the
returns by depth, the nearest return (the depth-back anchor), what the 1.5 m window and the 10 m camera check keep, and which
global-id 3D boxes the kept and the 6-10 m returns fall in. Writes a figure: photo, mask outline, returns coloured by depth,
kept returns ringed red, the anchor starred."""
from PIL import Image                     # PIL BEFORE aru_nerf_interface (libtiff/libjpeg soname clash otherwise)
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'
CASES = [(3112, 1617), (3113, 1632), (3115, 1660)]          # (keyframe, centre-tree detection), clip 15
DEPTH_BACK, MAX_DIST, W, H = 1.5, 10.0, 1280, 720
rig = load_rig(str(S)); K = rig['intrinsics']
proj = make_laser_projector(S, {'fl_x': K['fx'], 'fl_y': K['fy'], 'cx': K['cx'], 'cy': K['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S)
fe = json.load(open(G / 'clip_015/frame_entries.json')); m2g = json.load(open(G / 'global_ids.json'))
st = {int(k): v for k, v in m2g['stats'].items()}; m2g = m2g['member_to_global']
row_of = {o['id']: o['row_id'] for o in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
def boxes(P):
    out = {}
    for gid, s in st.items():
        n = int(np.all((P >= np.array(s['world_bbox_min'])) & (P <= np.array(s['world_bbox_max'])), axis=1).sum())
        if n: out[gid] = n
    return dict(sorted(out.items(), key=lambda t: -t[1])[:3])
fig, axs = plt.subplots(len(CASES), 1, figsize=(13, 7.6 * len(CASES)), dpi=90)
for ax, (kf, oid) in zip(np.atleast_1d(axs), CASES):
    ents = fe['frame_entries'][str(kf - 3000)]
    dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area'])) for e in ents]
    own = np.full((H, W), -1, np.int32)
    for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
    j = [i for i, d in enumerate(dec) if d[0] == oid][0]; owned = own == j
    uv, world, z = proj.project(int(kf)); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); cam = poses[kf][:3, 3]
    ins = owned[v, u]; zi, wi, ui, vi = z[ins], world[ins], u[ins], v[ins]
    a = int(np.argmin(zi)); dmin = float(zi[a]); keep = zi <= dmin + DEPTH_BACK; keep &= np.linalg.norm(wi - cam, axis=1) <= MAX_DIST
    far = (zi >= 6) & (zi <= 10)
    print(f'kf_{kf} oid {oid} (gid {m2g.get(f"15:{oid}")}): {int(owned.sum())} owned px, {len(z)} returns in the frame, {int(ins.sum())} inside the mask')
    print(f'   depth of in-mask returns: <4.5 m {int((zi < 4.5).sum())}, 4.5-6 m {int(((zi >= 4.5) & (zi < 6)).sum())}, 6-10 m {int(far.sum())}, >10 m {int((zi > 10).sum())}')
    print(f'   nearest return {dmin:.2f} m at pixel ({ui[a]}, {vi[a]}) -> window keeps <= {dmin + DEPTH_BACK:.2f} m: {int(keep.sum())} returns; 3D boxes of kept {boxes(wi[keep])}; of the 6-10 m returns {boxes(wi[far])}')
    ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{kf:06d}.png').convert('RGB'))
    ax.imshow(ph); ax.contour(owned, levels=[0.5], colors=['white'], linewidths=1.2)
    out = ~ins; ax.scatter(u[out], v[out], s=2, c='#999999', alpha=0.35, linewidths=0)
    sc = ax.scatter(ui, vi, s=7, c=zi, cmap='viridis', vmin=2, vmax=12, linewidths=0)
    ax.scatter(ui[keep], vi[keep], s=40, facecolors='none', edgecolors='red', linewidths=1.2)
    ax.scatter([ui[a]], [vi[a]], s=260, marker='*', c='red', edgecolors='white', linewidths=0.8)
    ax.set_title(f'kf_{kf}: SAM3 mask oid {oid} (white), {int(ins.sum())} laser returns inside it (coloured by depth, m); '
                 f'{int(far.sum())} at 6-10 m on the tree. Nearest {dmin:.1f} m (star) -> window keeps {int(keep.sum())} (red rings) -> '
                 + (f'id {m2g.get(f"15:{oid}")}' if m2g.get(f'15:{oid}') is not None else 'no id (coverage veto < 25)'), fontsize=10, loc='left')
    ax.set_axis_off(); plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
fig.tight_layout(); fn = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar/row17_lidar_in_mask_003112_003113_003115.png'); fig.savefig(fn); print('[fig]', fn)
