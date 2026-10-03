#!/usr/bin/env python3
"""18 anchor_counterfactual.py (nerf_new env): would #1617 / #1626 on kf_003112 get a tree id (and so a row) if the lifting window
started at the bulk of the returns instead of the single nearest one? (UJAMAA 2026-10-03; Paul: "why is this mask not in the row").
Same 1.5 m window, 10 m camera check and 25-point minimum; the tree = plurality of the kept returns over the lifting cluster 3D
boxes (global_ids.json stats; a proxy for the voxel-cluster labels, which need the whole-survey clustering). Test only."""
from PIL import Image
import json, sys, numpy as np
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
import survey_paths
from pathlib import Path
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720; K = 3112
I = load_rig(str(S))['intrinsics']; proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S); cam = poses[K][:3, 3]; st = {int(k): v for k, v in json.load(open(G / 'global_ids.json'))['stats'].items()}
row_of = {o['id']: o['row_id'] for o in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
ents = json.load(open(G / 'clip_015/frame_entries.json'))['frame_entries'][str(K % 200)]
dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area'])) for e in ents]
own = np.full((H, W), -1, np.int32)
for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
uv, world, z = proj.project(K); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); dist = np.linalg.norm(world - cam, axis=1)
def plural(P):
    c = {g: int(np.all((P >= np.array(s['world_bbox_min'])) & (P <= np.array(s['world_bbox_max'])), axis=1).sum()) for g, s in st.items()}
    c = {g: n for g, n in c.items() if n}; return sorted(c.items(), key=lambda t: -t[1])[:3]
for oid, region in ((1617, 'owned'), (1626, 'owned'), (1626, 'full')):
    j = [i for i, d in enumerate(dec) if d[0] == oid][0]; msk = (own == j) if region == 'owned' else dec[j][1]
    sel = msk[v, u]; zz, P, dd = z[sel], world[sel], dist[sel]
    for name, a in (('nearest return (current rule)', float(zz.min())), ('5th-percentile depth', float(np.percentile(zz, 5)))):
        keep = (zz <= a + 1.5) & (dd <= 10.0); top = plural(P[keep])
        g = top[0][0] if top and int(keep.sum()) >= 25 else None
        print(f'#{oid} {region} ({int(sel.sum())} returns): window from {name} {a:.2f} m keeps {int(keep.sum())}; boxes {top} -> ' + (f'tree {g}, row {row_of.get(g)}' if g is not None else 'no id'))
