#!/usr/bin/env python3
"""super_localise.py (2026-10-20) — is a part node localisable in 3D? For every part node of a seed, the Gaussians carrying it, their median
position in the survey (LIO) frame (the chain of method_fig_dump.py: run frame -> dataparser^-1 -> z-up -> R_W^-1), the horizontal distance of
that centroid to the tree's scene-graph marker, the same for the tree's own canopy Gaussians, the vertical offset trunk-vs-canopy (y down) and
the horizontal spread (median radius) of the part's Gaussians. Writes <out>.json and prints quartiles.
nerf_new pixi env:  super_localise.py <identity dir> <labels.npy> <nodes.npz> <out json> [--min 5]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('identity'); ap.add_argument('labels'); ap.add_argument('nodes'); ap.add_argument('out'); ap.add_argument('--min', type=int, default=5); ap.add_argument('--config', default='', help='splat config.yml to read the Gaussians from (default: the identity dir W.json config)'); a = ap.parse_args()
N = Path(a.identity); W = str(N).split('/experimental/')[0]; cfg = Path(a.config) if a.config else Path(json.load(open(N / 'W.json'))['config']); dp = json.load(open(cfg.parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); S = float(dp['scale'])
meta = json.load(open(W + '/experimental/h3dgs_expo/export_meta.json')); R_W = np.array(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], dtype=np.float64)
ck = sorted((cfg.parent / 'nerfstudio_models').glob('step-*.ckpt'))[-1]; sd = torch.load(ck, map_location='cpu', weights_only=False)['pipeline']; means = [v for k, v in sd.items() if k.endswith('gauss_params.means')][0].double().numpy()
n = means.shape[0]; p_h = (np.linalg.inv(T) @ np.c_[means / S, np.ones(n)].T).T[:, :3]; xyz = (np.linalg.inv(R_W) @ np.c_[p_h, np.ones(n)].T).T[:, :3]
lab = np.load(a.labels); nz = np.load(a.nodes); H = json.load(open(W + '/prod/bateleur/scene_graph/marker_hierarchy.json')); objs = {int(o['id']): np.array(o['xyz']) for o in H['objects']}
g = lambda q: q[..., [0, 2]]; rows = []
for nid, t, ty in zip(nz['ids'], nz['tree'], nz['type']):
    nid, t = int(nid), int(t); st = lab == nid; sg = lab == t
    if st.sum() < a.min or t not in objs: continue
    ct, cg, mk = np.median(xyz[st], 0), np.median(xyz[sg], 0), objs[t]
    rows.append({'node': nid, 'tree': t, 'type': int(ty), 'n_part': int(st.sum()), 'n_tree': int(sg.sum()), 'part_vs_marker_m': float(np.linalg.norm(g(ct) - g(mk))), 'canopy_vs_marker_m': float(np.linalg.norm(g(cg) - g(mk))), 'part_vs_canopy_m': float(np.linalg.norm(g(ct) - g(cg))), 'part_below_canopy_m': float(ct[1] - cg[1]), 'part_spread_m': float(np.median(np.linalg.norm(g(xyz[st]) - g(ct), axis=1)))})
q = lambda k: np.percentile([x[k] for x in rows], [25, 50, 75]).round(2) if rows else None
print(f"[loc] {len(rows)} part nodes with >= {a.min} gaussians (of {len(nz['ids'])}); gaussians per node median {np.median([x['n_part'] for x in rows]):.0f}")
print(f"[loc] part centroid to marker (horizontal) quartiles {q('part_vs_marker_m')} m; canopy centroid to marker {q('canopy_vs_marker_m')} m; part below canopy {q('part_below_canopy_m')} m; spread {q('part_spread_m')} m")
print(f"[loc] within 0.5 m of the marker: {sum(1 for x in rows if x['part_vs_marker_m'] <= 0.5)}/{len(rows)}; within 1.0 m: {sum(1 for x in rows if x['part_vs_marker_m'] <= 1.0)}/{len(rows)}; spread <= 0.5 m: {sum(1 for x in rows if x['part_spread_m'] <= 0.5)}/{len(rows)}")
json.dump(rows, open(a.out, 'w'), indent=1)
