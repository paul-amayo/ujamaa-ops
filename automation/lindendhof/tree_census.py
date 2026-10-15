#!/usr/bin/env python3
"""Lindendhof identity, step 3 (2026-10-12): joint census of the trunk observations of all lanes (one LO world frame) -> trees + rows.
Observations within MAX_RANGE m of their camera (depth error grows with range) are clustered in XY (DBSCAN, eps EPS m, >= MIN_OBS
observations from >= 2 frames); a tree = cluster median. Rows: the row axis = the dominant direction of the lane trajectories (the
lanes run along the rows); trees are binned by their across-row coordinate (1-D clustering with a gap of ROW_GAP m); each row gets a
line fit. Writes <out>/tree_census.json (trees: id, xyz, row_id, n_obs, n_frames, lanes; rows: id, offset, direction, n_trees) +
<out>/obs_labels.json (observation -> tree id or -1) + a top-down figure. Prints pitch / row-spacing statistics (the checks).
system python3 (numpy, sklearn):  tree_census.py <survey root> <trunk mask root> <out dir> [eps=0.35] [min_obs=5] [max_range=8]"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
S, M, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
EPS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35; MIN_OBS = int(sys.argv[5]) if len(sys.argv) > 5 else 5; MAX_RANGE = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0; ROW_GAP = 1.0
obs = []
for lane in sorted(M.glob('lane*')):
    for o in json.load(open(lane / 'trunk_obs.json')): o['lane'] = lane.name; obs.append(o)
use = [o for o in obs if o['range_m'] <= MAX_RANGE]; P = np.array([o['xyz'] for o in use])
# row axis from the lane cameras (lanes run along the rows): principal direction of all lane camera centres, per lane, averaged
dirs = []
for lane in sorted(S.glob('experimental/lane*/transforms_ref_lo.json')):
    C = np.array([np.array(f['transform_matrix'])[:3, 3] for f in json.load(open(lane))['frames']])[:, :2]
    u, s_, vt = np.linalg.svd(C - C.mean(0)); d = vt[0] / np.linalg.norm(vt[0]); dirs.append(d if (not dirs or d @ dirs[0] >= 0) else -d)
ax = np.mean(dirs, 0); ax /= np.linalg.norm(ax); perp = np.array([-ax[1], ax[0]])
lab = DBSCAN(eps=EPS, min_samples=MIN_OBS).fit_predict(P[:, :2]); trees = []
for k in sorted(set(lab) - {-1}):
    m = lab == k; frames = {use[i]['frame'] + use[i]['lane'] for i in np.nonzero(m)[0]}
    if len(frames) < 2: lab[m] = -1; continue
    trees.append({'cid': int(k), 'xyz': np.median(P[m], 0).round(3).tolist(), 'n_obs': int(m.sum()), 'n_frames': len(frames), 'lanes': sorted({use[i]['lane'] for i in np.nonzero(m)[0]}),
                  'spread_m': round(float(np.median(np.linalg.norm(P[m, :2] - np.median(P[m, :2], 0), axis=1))), 3), 'z_low': round(float(np.median([use[i]['z_low'] for i in np.nonzero(m)[0]])), 3)})
X = np.array([t['xyz'][:2] for t in trees]); a = X @ ax; c = X @ perp
order = np.argsort(c); rows = []; cur = [order[0]]
for i in order[1:]:
    if c[i] - c[cur[-1]] > ROW_GAP: rows.append(cur); cur = [i]
    else: cur.append(i)
rows.append(cur); rows = [r for r in rows if len(r) >= 3]
tid = 0; out_rows = []; cid_to_tid = {}
for ri, r in enumerate(sorted(rows, key=lambda r: np.median(c[r]))):
    r = sorted(r, key=lambda i: a[i]); out_rows.append({'id': ri, 'offset_m': round(float(np.median(c[r])), 3), 'direction': ax.round(5).tolist(), 'n_trees': len(r),
                                                        'pitch_m_median': round(float(np.median(np.diff(a[r]))), 3) if len(r) > 1 else None})
    for i in r: trees[i]['row_id'] = ri; trees[i]['id'] = tid; cid_to_tid[trees[i]['cid']] = tid; tid += 1
trees = [t for t in trees if 'id' in t]
labels = [{'lane': o['lane'], 'frame': o['frame'], 'idx': o['idx'], 'tree': -1} for o in obs]
key = {(o['lane'], o['frame'], o['idx']): j for j, o in enumerate(obs)}
for i, o in enumerate(use):
    if lab[i] >= 0 and int(lab[i]) in cid_to_tid: labels[key[(o['lane'], o['frame'], o['idx'])]]['tree'] = cid_to_tid[int(lab[i])]
json.dump({'source': f'lindendhof trunk census (tree_census.py): {len(use)} trunk observations within {MAX_RANGE} m, DBSCAN eps {EPS} m / {MIN_OBS} obs, rows by {ROW_GAP} m across-row gaps',
           'row_axis': ax.round(5).tolist(), 'trees': trees, 'rows': out_rows}, open(OUT / 'tree_census.json', 'w'), indent=1)
json.dump(labels, open(OUT / 'obs_labels.json', 'w'))
sp = np.diff(sorted(r['offset_m'] for r in out_rows)); pit = [r['pitch_m_median'] for r in out_rows if r['pitch_m_median']]
print(f'[census] {len(obs)} observations, {len(use)} within {MAX_RANGE} m -> {len(trees)} trees ({sum(1 for l in labels if l["tree"] >= 0)} observations assigned, '
      f'{int((lab == -1).sum())} noise); obs/tree median {np.median([t["n_obs"] for t in trees]):.0f}, spread median {np.median([t["spread_m"] for t in trees]):.3f} m')
print(f'[census] {len(out_rows)} rows: trees per row {[r["n_trees"] for r in out_rows]}; row spacing {sp.round(2).tolist()} m; in-row pitch median per row {pit}')
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axp = plt.subplots(figsize=(9, 9)); axp.scatter(P[:, 0], P[:, 1], s=1, c='0.8')
    for t in trees: axp.plot(t['xyz'][0], t['xyz'][1], 'o', ms=4, color=plt.cm.tab20(t['row_id'] % 20))
    for lane in sorted(S.glob('experimental/lane*/transforms_ref_lo.json')):
        C = np.array([np.array(f['transform_matrix'])[:3, 3] for f in json.load(open(lane))['frames']]); axp.plot(C[:, 0], C[:, 1], 'k-', lw=1)
    axp.set_aspect('equal'); axp.set_title(f'Lindendhof trunk census: {len(trees)} trees, {len(out_rows)} rows (grey = observations, black = lanes)'); fig.savefig(OUT / 'tree_census_topdown.png', dpi=110, bbox_inches='tight')
except Exception as e: print('[census] figure skipped:', e)
