#!/usr/bin/env python3
"""Compare global-id lifting runs (UJAMAA 2026-10-03; Paul: "use median points not closest points"). Runs: prod 01
global_ids.json (08-22, nearest-return anchor), the same rule rerun today, and the median anchor. Trees (clusters) are matched
across runs by 3D centroid (mutual nearest within 1.5 m) and named by their PROD id. Reports: trees found and matched,
detections with an id, detections that gain / lose / change tree vs the same-code nearest-return rerun (with SAM3 pixel area),
and the masks examined on kf_003112-3115."""
import json, sys
from pathlib import Path
import numpy as np
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; X = S / 'experimental/lifting_anchor_20261003'
RUNS = {'prod 08-22 (nearest)': G / 'global_ids.json', 'rerun (nearest)': X / 'min/global_ids.json', 'median': X / 'median/global_ids.json'}
J = {k: json.load(open(p)) for k, p in RUNS.items() if p.exists()}
area = {}
for c in json.load(open(G / 'clips.json'))['clips']:
    fe = json.load(open(G / f"clip_{c['clip_id']:03d}/frame_entries.json"))
    for li, ents in fe['frame_entries'].items():
        for e in ents: area[f"{c['clip_id']}:{int(e['obj_id'])}"] = (int(e.get('area', 0)), int(fe['frames'][int(li)]['kf_idx']))
P0 = J['prod 08-22 (nearest)']; C0 = {int(g): np.array(s['world_centroid']) for g, s in P0['stats'].items()}
def to_prod(run):
    C = {int(g): np.array(s['world_centroid']) for g, s in run['stats'].items()}
    a = np.array(list(C.values())); ga = list(C); b = np.array(list(C0.values())); gb = list(C0)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2); i_ab = d.argmin(1); i_ba = d.argmin(0)
    return {ga[i]: gb[i_ab[i]] for i in range(len(ga)) if i_ba[i_ab[i]] == i and d[i, i_ab[i]] <= 1.5}
for k, r in J.items():
    mp = to_prod(r); m2g = r['member_to_global']
    print(f'{k:22s}: {r["n_global_ids"]} trees ({len(mp)} matched to a prod tree within 1.5 m), {len(m2g)} detections with an id '
          f'({sum(area.get(m, (0, 0))[0] for m in m2g) / 1e6:.1f} M SAM3 px), anchor {r.get("depth_anchor", "min (pre-flag)")}')
if 'median' in J and 'rerun (nearest)' in J:
    A, B = J['rerun (nearest)'], J['median']; ma, mb = to_prod(A), to_prod(B)
    na = {m: ma.get(g, ('unmatched', g)) for m, g in A['member_to_global'].items()}; nb = {m: mb.get(g, ('unmatched', g)) for m, g in B['member_to_global'].items()}
    gain = [m for m in nb if m not in na]; lose = [m for m in na if m not in nb]; chg = [m for m in nb if m in na and nb[m] != na[m]]
    px = lambda L: sum(area.get(m, (0, 0))[0] for m in L) / 1e6
    print(f'median vs nearest (same code): gain an id {len(gain)} ({px(gain):.1f} M px), lose an id {len(lose)} ({px(lose):.1f} M px), '
          f'change tree {len(chg)} ({px(chg):.1f} M px), same {sum(1 for m in nb if m in na and nb[m] == na[m])}')
    big_gain = sorted(gain, key=lambda m: -area.get(m, (0, 0))[0])[:8]
    print('   largest masks that gain an id:', [(m, area[m][1], area[m][0], nb[m]) for m in big_gain])
    big_chg = sorted(chg, key=lambda m: -area.get(m, (0, 0))[0])[:8]
    print('   largest masks that change tree (key, kf, px, nearest -> median):', [(m, area[m][1], area[m][0], na[m], nb[m]) for m in big_chg])
    for m in ('15:1617', '15:1618', '15:1626', '15:1632', '15:1633', '15:1660', '15:1661', '15:1669'):
        print(f'   {m} (kf {area[m][1]}, {area[m][0]} px): prod {P0["member_to_global"].get(m)}, nearest rerun {na.get(m)}, median {nb.get(m)}')
