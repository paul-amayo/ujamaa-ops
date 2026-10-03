#!/usr/bin/env python3
"""8 chunk_void_scan.py (nerf_new env): how often does the lifting's nearest-return window blank a tree the LiDAR sees?
(UJAMAA 2026-10-03; Paul: "we missed the closest tree"). Replays cluster_tree_instances.collect() per keyframe of 01 chunk 3_1:
partition hygiene (smallest-on-top owned pixels, drop masks whose remainder is not ONE >=500 px blob), the LaserProjector,
the 1.5 m depth-back window from the mask's nearest return, the 10 m camera check, the ground filter (0.3 m above the nearest
trajectory height) and the coverage veto (<25 pts or <0.5 pts/kpx of SAM3 area). Per detection: returns in its owned pixels
within 10 m (what the LiDAR saw) vs returns the window keeps, and its final global id. Class 'window' = the LiDAR saw >= 25
non-ground returns within 10 m but the window kept < veto -> no id. Usage: chunk_void_scan.py [K ...] (default: chunk frames)."""
from PIL import Image                     # PIL BEFORE aru_nerf_interface
import json, sys, glob, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import survey_paths
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; NB = S / 'experimental/h3dgs_native/chunk_3_1_sam3'
DB, MD, GM, W, H = 1.5, 10.0, 0.3, 1280, 720
rig = load_rig(str(S)); K_ = rig['intrinsics']
proj = make_laser_projector(S, {'fl_x': K_['fx'], 'fl_y': K_['fy'], 'cx': K_['cx'], 'cy': K_['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S); traj = np.array([T[:3, 3] for T in poses.values()]); gt = cKDTree(traj[:, [0, 2]])
m2g = json.load(open(G / 'global_ids.json'))['member_to_global']
frames = [int(a) for a in sys.argv[1:]] or sorted(int(os.path.basename(f)[3:9]) for f in glob.glob(str(NB / 'supervision/trees_only/kf_*.png')))
clips = {}
rows, t0 = [], time.time()
for n, K in enumerate(frames):
    cid, li = K // 200, K % 200
    if cid not in clips: clips = {cid: json.load(open(G / f'clip_{cid:03d}/frame_entries.json'))}
    ents = clips[cid]['frame_entries'].get(str(li), [])
    if not ents or K not in poses: continue
    dec = []
    for e in ents:
        m = mu.decode(e['rle']).astype(bool)
        if m.shape == (H, W): dec.append((int(e['obj_id']), m, int(e['area'])))
    own = np.full((H, W), -1, np.int32)
    for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
    uv, world, z = proj.project(int(K)); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); cam = poses[K][:3, 3]
    dist = np.linalg.norm(world - cam, axis=1); _, nn = gt.query(world[:, [0, 2]]); ground = world[:, 1] >= traj[nn, 1] - GM
    lab_own = own[v, u]
    for j, (oid, m, area) in enumerate(dec):
        o = own == j; gid = m2g.get(f'{cid}:{oid}'); n_own = int(o.sum())
        if n_own == 0: rows.append((K, oid, area, n_own, 0, 0, 0, None, gid, 'absorbed')); continue
        lab, _ = ndimage.label(o); sizes = np.bincount(lab.ravel())[1:]
        if int((sizes >= 500).sum()) != 1: rows.append((K, oid, area, n_own, 0, 0, 0, None, gid, 'hygiene')); continue
        ins = lab_own == j
        seen = int((ins & (dist <= MD) & ~ground).sum())
        if not ins.any(): rows.append((K, oid, area, n_own, 0, seen, 0, None, gid, 'id' if gid is not None else 'no returns')); continue
        dmin = float(z[ins].min()); kept = int((ins & (z <= dmin + DB) & (dist <= MD) & ~ground).sum())
        veto = kept < 25 or 1000.0 * kept / max(area, 1) < 0.5
        cls = 'id' if gid is not None else ('window' if (veto and seen >= 25 and 1000.0 * seen / max(area, 1) >= 0.5) else ('sparse' if veto else 'other'))
        rows.append((K, oid, area, n_own, int(ins.sum()), seen, kept, round(dmin, 2), gid, cls))
    if n % 100 == 0: print(f'[scan] {n}/{len(frames)} frames, {time.time() - t0:.0f} s', flush=True)
out = NB / 'void_scan.json'; json.dump(rows, open(out, 'w'))
c = Counter(r[9] for r in rows); px = defaultdict(int)
for r in rows: px[r[9]] += r[3]
print(f'[scan] {len(frames)} frames, {len(rows)} detections -> {out}')
for k in ('id', 'window', 'sparse', 'other', 'no returns', 'hygiene', 'absorbed'):
    print(f'   {k:10s}: {c[k]:6d} detections, {px[k] / 1e6:7.1f} M owned px')
big = [r for r in rows if r[9] == 'window' and r[3] >= 10000]
print(f'   window-blanked detections owning >= 10k px: {len(big)} on {len({r[0] for r in big})} frames; median LiDAR-seen returns {int(np.median([r[5] for r in big])) if big else 0}, median kept {int(np.median([r[6] for r in big])) if big else 0}')
