#!/usr/bin/env python3
"""Lindendhof identity, step 3 v2 (2026-10-12): the trees stand densely along narrow rows, so 2-D DBSCAN chains a whole row into one
cluster (v1: 169 'trees', one 'row' of 123). Rows and trees are found as 1-D density peaks instead:
  rows  = peaks of the across-row histogram of all LiDAR trunk observations (bin 0.05 m, Gaussian sigma 0.1 m, peaks >= ROW_SEP apart);
          an observation belongs to a row if it lies within ROW_HALF m of the peak
  trees = per row, peaks of the along-row histogram (bin 0.02 m, sigma SIGMA m, peaks >= MIN_SEP apart, height >= MIN_OBS
          observations within +-0.15 m of the peak, seen from >= 2 frames); an observation belongs to the nearest peak within ASSIGN m
Row axis = the mean direction of the lane camera tracks. Writes tree_census.json (trees: id, xyz, row_id, n_obs, n_frames, lanes,
spread_m; rows: id, offset_m, direction, n_trees, pitch_m_median), obs_labels.json, tree_census_topdown.png + a pitch histogram.
nerf_new pixi env:  tree_census_v2.py <survey root> <trunk mask root> <out dir> [min_sep=0.3] [sigma=0.06] [min_obs=5] [max_range=8]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
S, M, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
MIN_SEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3; SIGMA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.06
MIN_OBS = int(sys.argv[6]) if len(sys.argv) > 6 else 5; MAX_RANGE = float(sys.argv[7]) if len(sys.argv) > 7 else 8.0; ROW_SEP, ROW_HALF, ASSIGN = float(__import__("os").environ.get("ROW_SEP", "1.5")), 0.5, 0.25   # ROW_SEP env (2026-10-12: rows are ~3 m apart; 2.4 forbids split row peaks)
obs = []
for lane in sorted(M.glob('lane*')):
    for o in json.load(open(lane / 'trunk_obs.json')): o['lane'] = lane.name; obs.append(o)
use = [o for o in obs if o['range_m'] <= MAX_RANGE]; P = np.array([o['xyz'] for o in use])
dirs = []
for lane in sorted(S.glob('experimental/lane*/transforms_ref_lo.json')):
    C = np.array([np.array(f['transform_matrix'])[:3, 3] for f in json.load(open(lane))['frames']])[:, :2]
    _, _, vt = np.linalg.svd(C - C.mean(0)); d = vt[0]; dirs.append(d if (not dirs or d @ dirs[0] >= 0) else -d)
ax = np.mean(dirs, 0); ax /= np.linalg.norm(ax); perp = np.array([-ax[1], ax[0]]); a, c = P[:, :2] @ ax, P[:, :2] @ perp
def peaks(x, bin_m, sigma_m, sep_m, height):
    lo, hi = x.min() - 1, x.max() + 1; h, e = np.histogram(x, bins=np.arange(lo, hi + bin_m, bin_m)); hs = gaussian_filter1d(h.astype(float), sigma_m / bin_m)
    pk, _ = find_peaks(hs, distance=max(1, int(sep_m / bin_m)), height=height); return (e[pk] + e[pk + 1]) / 2
rows_c = peaks(c, 0.05, 0.1, ROW_SEP, height=20)
row_of = np.full(len(use), -1); dist = np.abs(c[:, None] - rows_c[None, :]); nr = dist.argmin(1); row_of[dist[np.arange(len(use)), nr] <= ROW_HALF] = nr[dist[np.arange(len(use)), nr] <= ROW_HALF]
trees, tree_of, out_rows, tid = [], np.full(len(use), -1), [], 0
for ri in np.argsort(rows_c):
    m = np.nonzero(row_of == ri)[0]
    if len(m) < MIN_OBS: continue
    pk = peaks(a[m], 0.02, SIGMA, MIN_SEP, height=0)
    d = np.abs(a[m][:, None] - pk[None, :]); near = d.argmin(1); ok = d[np.arange(len(m)), near] <= ASSIGN
    row_trees = []
    for k, p in enumerate(pk):
        mm = m[ok & (near == k)]; core = mm[np.abs(a[mm] - p) <= 0.15]; frames = {use[i]['lane'] + use[i]['frame'] for i in mm}
        if len(core) < MIN_OBS or len(frames) < 2: continue
        xyz = np.median(P[mm], 0); row_trees.append((p, mm, xyz, frames))
    if len(row_trees) < 3: continue
    row_id = len(out_rows)
    for p, mm, xyz, frames in row_trees:
        trees.append({'id': tid, 'row_id': row_id, 'xyz': xyz.round(3).tolist(), 'along_m': round(float(p), 3), 'n_obs': int(len(mm)), 'n_frames': len(frames),
                      'lanes': sorted({use[i]['lane'] for i in mm}), 'spread_m': round(float(np.median(np.abs(a[mm] - p))), 3), 'z_low': round(float(np.median([use[i]['z_low'] for i in mm])), 3)})
        tree_of[mm] = tid; tid += 1
    al = np.array([t[0] for t in row_trees])
    out_rows.append({'id': row_id, 'offset_m': round(float(rows_c[ri]), 3), 'direction': ax.round(5).tolist(), 'n_trees': len(row_trees),
                     'pitch_m_median': round(float(np.median(np.diff(al))), 3), 'pitch_m_p10_p90': np.percentile(np.diff(al), [10, 90]).round(3).tolist(), 'length_m': round(float(al.max() - al.min()), 2)})
labels = [{'lane': o['lane'], 'frame': o['frame'], 'idx': o['idx'], 'tree': -1} for o in obs]; key = {(o['lane'], o['frame'], o['idx']): j for j, o in enumerate(obs)}
for i, o in enumerate(use):
    if tree_of[i] >= 0: labels[key[(o['lane'], o['frame'], o['idx'])]]['tree'] = int(tree_of[i])
json.dump({'source': f'lindendhof trunk census v2 (tree_census_v2.py): {len(use)} LiDAR trunk observations within {MAX_RANGE} m; rows = across-row density peaks (>= {ROW_SEP} m apart, +-{ROW_HALF} m), trees = along-row density peaks (>= {MIN_SEP} m apart, sigma {SIGMA} m, >= {MIN_OBS} obs within 0.15 m, >= 2 frames)',
           'row_axis': ax.round(5).tolist(), 'trees': trees, 'rows': out_rows}, open(OUT / 'tree_census.json', 'w'), indent=1)
json.dump(labels, open(OUT / 'obs_labels.json', 'w'))
sp = np.diff(sorted(r['offset_m'] for r in out_rows))
print(f'[census-v2] {len(obs)} observations, {len(use)} within {MAX_RANGE} m, {int((row_of >= 0).sum())} on {len(rows_c)} row peaks -> {len(out_rows)} rows, {len(trees)} trees '
      f'({int((tree_of >= 0).sum())} observations assigned); obs/tree median {np.median([t["n_obs"] for t in trees]):.0f}, spread median {np.median([t["spread_m"] for t in trees]):.3f} m')
print(f'[census-v2] rows: trees {[r["n_trees"] for r in out_rows]}; length m {[r["length_m"] for r in out_rows]}; row spacing {sp.round(2).tolist()} m; '
      f'pitch median {[r["pitch_m_median"] for r in out_rows]}; p10/p90 {[r["pitch_m_p10_p90"] for r in out_rows]}')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, (axp, axh) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [2, 1]}); axp.scatter(P[:, 0], P[:, 1], s=1, c='0.82')
for t in trees: axp.plot(t['xyz'][0], t['xyz'][1], 'o', ms=3, color=plt.cm.tab10(t['row_id'] % 10))
for lane in sorted(S.glob('experimental/lane*/transforms_ref_lo.json')):
    C = np.array([np.array(f['transform_matrix'])[:3, 3] for f in json.load(open(lane))['frames']]); axp.plot(C[:, 0], C[:, 1], 'k-', lw=1)
axp.set_aspect('equal'); axp.set_title(f'{len(trees)} trees in {len(out_rows)} rows (grey = LiDAR trunk observations, black = lanes)')
pt = np.concatenate([np.diff(sorted(t['along_m'] for t in trees if t['row_id'] == r['id'])) for r in out_rows]) if out_rows else np.zeros(0)
axh.hist(pt, bins=np.arange(0, 3.0, 0.05), color='0.4'); axh.set_xlabel('in-row spacing between consecutive trees (m)'); axh.set_ylabel('count'); axh.set_title('tree pitch')
fig.savefig(OUT / 'tree_census_topdown.png', dpi=100, bbox_inches='tight')
