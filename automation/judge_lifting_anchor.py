#!/usr/bin/env python3
"""Judge the masks whose tree id differs between the nearest-return and median lifting anchors (UJAMAA 2026-10-03). For every
mask that LOSES its id or CHANGES tree under the median rule: project the LiDAR into its owned pixels (the lifting's own
projector + partition), and ask which 3D tree box (prod cluster boxes, runs matched to prod ids by centroid) holds the MOST
of its returns within 10 m, and how much of the mask lies beyond 10 m. A change is "right" when the median's tree is that
majority tree; a loss is a correction when the mask's own returns are mostly beyond 10 m (its tree is out of range, so the
nearest-return id came from a minority of nearer returns). nerf_new env."""
from PIL import Image
import json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/automation')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
import survey_paths
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; X = S / 'experimental/lifting_anchor_20261003'; W, H = 1280, 720
P0 = json.load(open(G / 'global_ids.json')); A = json.load(open(X / 'min/global_ids.json')); B = json.load(open(X / 'median/global_ids.json'))
C0 = {int(g): np.array(s['world_centroid']) for g, s in P0['stats'].items()}
def to_prod(run):
    C = {int(g): np.array(s['world_centroid']) for g, s in run['stats'].items()}; a = np.array(list(C.values())); ga = list(C); b = np.array(list(C0.values())); gb = list(C0)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2); i_ab = d.argmin(1); i_ba = d.argmin(0)
    return {ga[i]: gb[i_ab[i]] for i in range(len(ga)) if i_ba[i_ab[i]] == i and d[i, i_ab[i]] <= 1.5}
ma, mb = to_prod(A), to_prod(B)
na = {m: ma.get(g) for m, g in A['member_to_global'].items()}; nb = {m: mb.get(g) for m, g in B['member_to_global'].items()}
lost = [m for m in na if m not in nb]; chg = [m for m in nb if m in na and nb[m] != na[m]]; gain = [m for m in nb if m not in na]
if '--gains' in sys.argv: lost, chg = [], gain
BOX = {int(g): (np.array(s['world_bbox_min']), np.array(s['world_bbox_max'])) for g, s in P0['stats'].items()}
I = load_rig(str(S))['intrinsics']; proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
poses = survey_paths.kf_pose_dict(S)
clips = json.load(open(G / 'clips.json'))['clips']; by_kf = defaultdict(list)
for m in lost + chg:
    cid, oid = map(int, m.split(':')); by_kf[(cid, oid)]
want = defaultdict(set)
for m in lost + chg: cid, oid = map(int, m.split(':')); want[cid].add(oid)
res = {}
for cid in sorted(want):
    fe = json.load(open(G / f'clip_{cid:03d}/frame_entries.json'))
    for li, ents in fe['frame_entries'].items():
        oids = {int(e['obj_id']) for e in ents}
        if not (oids & want[cid]): continue
        K = int(fe['frames'][int(li)]['kf_idx'])
        if K not in poses: continue
        dec = [(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area'])) for e in ents]
        own = np.full((H, W), -1, np.int32)
        for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j
        try: uv, world, z = proj.project(K)
        except Exception: continue
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); lab = own[v, u]; dist = np.linalg.norm(world - poses[K][:3, 3], axis=1)
        for j, (oid, m, area) in enumerate(dec):
            if oid not in want[cid]: continue
            sel = lab == j; Pw = world[sel]; dd = dist[sel]
            if not sel.any(): res[f'{cid}:{oid}'] = (K, 0, 0.0, None, 0.0); continue
            near = dd <= 10.0; cnt = Counter()
            for g, (lo, hi) in BOX.items():
                c = int(np.all((Pw[near] >= lo) & (Pw[near] <= hi), axis=1).sum())
                if c: cnt[g] = c
            maj, mc = (cnt.most_common(1)[0] if cnt else (None, 0))
            res[f'{cid}:{oid}'] = (K, int(sel.sum()), float((~near).mean()), maj, mc / max(int(near.sum()), 1))
if '--gains' in sys.argv:
    ok = sum(1 for m in chg if m in res and res[m][3] == nb[m]); far_ = sum(1 for m in chg if m in res and res[m][2] > 0.5)
    print(f'gained an id: {len(chg)} masks ({sum(1 for m in chg if m in res)} measured): the new id is the majority tree of the mask\'s returns within 10 m in {ok}; '
          f'another tree is the majority in {sum(1 for m in chg if m in res) - ok}; masks with most returns beyond 10 m {far_}')
    json.dump({m: [nb[m]] + list(res.get(m, ())) for m in chg}, open(X / 'judge_gains.json', 'w')); sys.exit(0)
chg_right = sum(1 for m in chg if m in res and res[m][3] == nb[m]); chg_wrong = sum(1 for m in chg if m in res and res[m][3] == na[m]); chg_neither = sum(1 for m in chg if m in res) - chg_right - chg_wrong
far = [m for m in lost if m in res and res[m][2] > 0.5]; lost_majority_kept = [m for m in lost if m in res and res[m][2] <= 0.5 and res[m][3] == na[m]]
print(f'changed tree: {len(chg)} masks ({sum(1 for m in chg if m in res)} measured): the median\'s tree holds most of the mask\'s returns within 10 m in {chg_right}, the nearest-return tree in {chg_wrong}, neither in {chg_neither}')
print(f'lost an id: {len(lost)} masks ({sum(1 for m in lost if m in res)} measured): most returns beyond 10 m (tree out of range; the old id came from nearer returns) {len(far)}; '
      f'mostly within 10 m AND the old id is the majority tree (a real loss) {len(lost_majority_kept)}; other {sum(1 for m in lost if m in res) - len(far) - len(lost_majority_kept)}')
json.dump({'changed': {m: [na[m], nb[m]] + list(res.get(m, ())) for m in chg}, 'lost': {m: [na[m]] + list(res.get(m, ())) for m in lost}},
          open(X / 'judge.json', 'w'))
print('->', X / 'judge.json')
