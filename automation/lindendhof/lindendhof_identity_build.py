#!/usr/bin/env python3
"""Lindendhof identity, steps 4-5 (2026-10-12, design approved by Paul): hierarchy + per-frame supervision from the trunk census.
  hierarchy  <out>/marker_hierarchy.json — objects = census trees (id, row_id, xyz), rows (id, object_ids), no fruit (the schema the
             graph embedder trained on for pepper: objects / rows / fruits / source / n_objects)
  supervision per lane, for every frame SAM3 ran on (trunk_masks/lane<n>): uint16 id map, 65535 = unlabelled:
    canopy  pixel's 3D point (the lane's v3 model depth, back-projected to the LO world) 0.5-4.0 m above the local ground (LiDAR 5th
            percentile on a 0.5 m grid), within ROW_HALF m of a census row -> the row's nearest tree along the row if within
            TREE_HALF m (the hedge is split between trunks in 3D, as approved)
    trunk   a SAM3 trunk mask whose base observation the census assigned to a tree -> that tree id (overrides the canopy rule)
  + manifest (supervision_manifest/v1 fields: word / level tables via HiGH get_word_for_id, unlabelled 65535) and split_names.json
    (census = the workspace's training frames with a map, eval = its held-out frames).
nerf_new pixi env:  lindendhof_identity_build.py <survey root> <identity dir> [lanes=1,2,3,4,5]"""
import datetime, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH'); from word_utils import get_word_for_id
S, ID = Path(sys.argv[1]), Path(sys.argv[2]); LANES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4,5').split(',')]
import os; ROW_HALF, TREE_HALF, ZLO, ZHI = (float(os.environ.get(k, d)) for k, d in (('ROW_HALF', '1.2'), ('TREE_HALF', '1.0'), ('CZLO', '0.5'), ('CZHI', '4.0'))) ; UNLAB = 65535   # env for other sites (Dec: bushes 0.25-2.0 m, rows 0.6, bags 0.6)
WSMAP = dict(kv.split(':') for kv in os.environ.get('WSMAP', '').split(',') if kv); MASKS = os.environ.get('MASKS', 'trunk_masks')
cen = json.load(open(ID / 'tree_census.json')); ax = np.array(cen['row_axis'], np.float64); perp = np.array([-ax[1], ax[0]])
trees = cen['trees']; rows = cen['rows']
H = {'objects': [{'id': t['id'], 'name': f"Object Object({t['id']})", 'row_id': t['row_id'], 'super_row_id': -1, 'section_id': -1, 'xyz': t['xyz']} for t in trees],
     'rows': [{'id': r['id'], 'object_ids': [t['id'] for t in trees if t['row_id'] == r['id']]} for r in rows], 'fruits': [],
     'source': f"{os.environ.get('SITE', 'Lindendhof')} anchor census ({cen['source'][:160]})", 'n_objects': len(trees)}
json.dump(H, open(ID / 'marker_hierarchy.json', 'w'), indent=1)
row_off = np.array([r['offset_m'] for r in rows]); row_trees = {r['id']: sorted([(t['along_m'], t['id']) for t in trees if t['row_id'] == r['id']]) for r in rows}
labels = json.load(open(ID / 'obs_labels.json')); assigned = {(l['lane'], l['frame'], l['idx']): l['tree'] for l in labels if l['tree'] >= 0}
def read_ply(p):
    with open(p, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'): n = int(line.split()[-1])
            if line == 'end_header': break
        a = np.frombuffer(f.read(n * 15), dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]), count=n)
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
for n in LANES:
    lane = f'lane{n}'; wsn = WSMAP.get(str(n), f'chunk_lane{n}'); cfg = S / f'experimental/v3/{wsn}/outputs/sf_60000_bilateral/splatfacto/run/config.yml'; OUT = ID / f'cells/{lane}'; SUP = OUT / 'supervision/trees_only'; SUP.mkdir(parents=True, exist_ok=True)
    P = read_ply(S / f'experimental/lane{n}/init_lidar.ply'); g0 = P[:, :2].min(0); gi = np.floor((P[:, :2] - g0) / 0.5).astype(int); shape = gi.max(0) + 1
    order = np.lexsort((P[:, 2], gi[:, 1], gi[:, 0])); grid = np.full(shape, np.nan); flat = gi[:, 0] * shape[1] + gi[:, 1]
    for cell in np.unique(flat):
        z = P[flat == cell, 2] if False else None
    zs = {}; fs = flat[order]; zz = P[order, 2]; bounds = np.flatnonzero(np.diff(fs)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(fs)]
    for s0, e0 in zip(starts, ends):
        if e0 - s0 >= 20: grid[fs[s0] // shape[1], fs[s0] % shape[1]] = zz[s0 + int(0.05 * (e0 - s0))]
    vi = np.argwhere(np.isfinite(grid)); vz = grid[np.isfinite(grid)]
    for ij in np.argwhere(~np.isfinite(grid)):   # fill empty cells with the nearest measured one
        grid[ij[0], ij[1]] = vz[np.argmin(((vi - ij) ** 2).sum(1))]
    _, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
    dp = json.load(open(cfg.parent / 'dataparser_transforms.json')); T = np.array(dp['transform'], np.float64); s = float(dp['scale']); RT, tT = T[:, :3], T[:, 3]
    cams = {}
    for ds in (pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset):
        for i, p in enumerate(ds.image_filenames): cams[Path(p).name] = (ds, i)
    split = json.load(open(S / f'experimental/v3/{wsn}/split.json')); present = set(); done = []
    for jf in sorted((ID / f'{MASKS}/{lane}').glob('f_*.json')):
        fname = jf.name.replace('.json', '.png')
        if fname not in cams: continue
        ds, i = cams[fname]; cam = ds.cameras[i:i + 1].to(dev)
        with torch.no_grad(): depth = model.get_outputs(cam)['depth'][..., 0].float().cpu().numpy()
        Hh, Ww = depth.shape; vv, uu = np.mgrid[0:Hh, 0:Ww]; fx, fy, cx, cy = (float(v) for v in (cam.fx[0, 0], cam.fy[0, 0], cam.cx[0, 0], cam.cy[0, 0]))
        c2w = cam.camera_to_worlds[0].cpu().numpy().astype(np.float64); z = depth.reshape(-1)
        pc = np.stack([(uu.reshape(-1) - cx) / fx * z, -(vv.reshape(-1) - cy) / fy * z, -z], 1); pw = ((pc @ c2w[:3, :3].T + c2w[:3, 3]) / s - tT) @ RT
        gij = np.clip(np.floor((pw[:, :2] - g0) / 0.5).astype(int), 0, shape - 1); hgt = pw[:, 2] - grid[gij[:, 0], gij[:, 1]]
        cc, aa = pw[:, :2] @ perp, pw[:, :2] @ ax; lab = np.full(Hh * Ww, UNLAB, np.uint16)
        canopy = np.isfinite(z) & (z > 0) & (hgt > ZLO) & (hgt < ZHI)
        dr = np.abs(cc[:, None] - row_off[None, :]); ri = dr.argmin(1); inrow = canopy & (dr[np.arange(len(ri)), ri] < ROW_HALF)
        for r in rows:
            sel = np.flatnonzero(inrow & (ri == rows.index(r)))
            if not len(sel) or not row_trees[r['id']]: continue
            al = np.array([x[0] for x in row_trees[r['id']]]); ids = np.array([x[1] for x in row_trees[r['id']]])
            k = np.clip(np.searchsorted(al, aa[sel]), 1, len(al) - 1); k = np.where(np.abs(aa[sel] - al[k - 1]) <= np.abs(aa[sel] - al[k]), k - 1, k)
            ok = np.abs(aa[sel] - al[k]) < TREE_HALF; lab[sel[ok]] = ids[k[ok]]
        lab = lab.reshape(Hh, Ww); tm = np.array(Image.open(jf.with_suffix('.png')))
        for it in json.load(open(jf)):
            t = assigned.get((lane, fname, it['idx']))
            if t is not None: lab[tm == it['idx']] = t
        Image.fromarray(lab, mode='I;16').save(SUP / fname); present |= set(np.unique(lab).tolist()); done.append(fname)
    present.discard(UNLAB)
    man = {'schema': 'supervision_manifest/v1', 'created': datetime.datetime.now().isoformat(timespec='seconds'), 'unlabelled_value': UNLAB, 'fruit_id_base': 10000,
           'filter_spec': {'name': 'lindendhof_trunk_canopy_v1', 'rule': f'canopy {ZLO}-{ZHI} m above ground within {ROW_HALF} m of a row -> nearest tree along the row (< {TREE_HALF} m); census-assigned SAM3 trunk masks override'},
           'word_table': {str(k): get_word_for_id(int(k), 'mask') for k in sorted(present)}, 'level_table': {str(k): 'tree' for k in sorted(present)},
           'tree_source': {'census': str(ID / 'tree_census.json'), 'depth': f'v3 {cfg}', 'trunk_masks': str(ID / f'{MASKS}/{lane}')},
           'hierarchy': {'path': str(ID / 'marker_hierarchy.json')}, 'counts': {'frames': len(done)}}
    json.dump(man, open(SUP / 'manifest.json', 'w'), indent=1)
    have = set(done); json.dump({'source': f'{lane} v3 workspace split x frames with maps', 'train': [f for f in split['train'] if f in have], 'eval': [f for f in split['test'] if f in have]}, open(OUT / 'split_names.json', 'w'), indent=0)
    print(f'[build] {lane}: {len(done)} maps ({sum(1 for f in split["train"] if f in have)} census / {sum(1 for f in split["test"] if f in have)} held-out), {len(present)} tree ids present', flush=True)
    del pipe, model; torch.cuda.empty_cache()
print(f"[build] hierarchy: {len(trees)} trees, {len(rows)} rows -> {ID / 'marker_hierarchy.json'}")
