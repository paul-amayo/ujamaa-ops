#!/usr/bin/env python3
"""parts_sheet.py (2026-10-20) — what did SAM 3 find for each prompt? From one or more parts_masks.py output dirs: per prompt the masks, views
with >= 1, trees with >= 3 masks (would-be part nodes), area as % of the frame (median / p90 / max), share of the parent tree, median score;
and a check sheet: one row per prompt, the top --n detections by score (one per view), each a crop around the mask with its outline, labelled
with the score and the area. system python3:  parts_sheet.py <out.png> <parts dir> [<parts dir> ...] [--images <dir>] [--n 6]"""
import argparse, json, glob, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage as nd
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--images', default=''); ap.add_argument('--n', type=int, default=6); a = ap.parse_args()
recs = collections.defaultdict(list)
for d in a.dirs:
    for f in sorted(glob.glob(f'{d}/kf_*.json')):
        for i, r in enumerate(json.load(open(f))): r['_dir'], r['_frame'], r['_i'] = d, Path(f).name.replace('.json', '.png'), i; recs[r['prompt']].append(r)
H, W = np.load(sorted(glob.glob(f'{a.dirs[0]}/kf_*.npz'))[0])['shape']; P = H * W; rows = []
for p, rs in recs.items():
    ar = np.array([r['area'] for r in rs]) / P * 100; trees = collections.Counter(r['tree'] for r in rs)
    rows.append({'prompt': p, 'masks': len(rs), 'views': len({r['_frame'] for r in rs}), 'trees_ge3': sum(1 for v in trees.values() if v >= 3), 'area_pct_median': round(float(np.median(ar)), 3), 'area_pct_p90': round(float(np.percentile(ar, 90)), 3), 'area_pct_max': round(float(ar.max()), 3), 'of_tree_median': round(float(np.median([r['frac_of_tree'] for r in rs])), 3), 'score_median': round(float(np.median([r['score'] for r in rs])), 3)})
    print(f"[sheet] {p:16s} masks {len(rs):4d} | views {rows[-1]['views']:3d} | trees with >= 3 masks {rows[-1]['trees_ge3']:2d} | area % of frame median {rows[-1]['area_pct_median']:.3f} p90 {rows[-1]['area_pct_p90']:.3f} max {rows[-1]['area_pct_max']:.2f} | of tree {rows[-1]['of_tree_median']:.3f} | score {rows[-1]['score_median']:.2f}")
json.dump(rows, open(str(a.out).replace('.png', '.json'), 'w'), indent=1)
IM = Path(a.images) if a.images else None
order = sorted(recs, key=lambda p: -len(recs[p])); fig, axes = plt.subplots(len(order), a.n, figsize=(a.n * 1.6, len(order) * 1.7), squeeze=False); fig.subplots_adjust(0.08, 0.005, 0.995, 0.99, 0.03, 0.18)
for ri, p in enumerate(order):
    seen, picks = set(), []
    for r in sorted(recs[p], key=lambda r: -r['score']):
        if r['_frame'] in seen: continue
        seen.add(r['_frame']); picks.append(r)
        if len(picks) == a.n: break
    axes[ri, 0].text(-0.08, 0.5, f"{p}\n{len(recs[p])} masks", transform=axes[ri, 0].transAxes, ha='right', va='center', fontsize=7)
    for ci in range(a.n):
        ax = axes[ri, ci]; ax.axis('off')
        if ci >= len(picks): continue
        r = picks[ci]; z = np.load(f"{r['_dir']}/{r['_frame'].replace('.png', '.npz')}"); m = np.unpackbits(z['masks'][r['_i']])[:P].reshape(H, W).astype(bool)
        img = np.array(Image.open(IM / r['_frame']).convert('RGB')) if IM else np.zeros((H, W, 3), np.uint8)
        x0, y0, x1, y1 = r['box']; s = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; h = max(int(0.8 * s), 40)
        X0, X1, Y0, Y1 = max(0, cx - h), min(W, cx + h), max(0, cy - h), min(H, cy + h); edge = m ^ nd.binary_erosion(m, iterations=2)
        im = img.copy(); im[edge] = [255, 0, 255]; ax.imshow(im[Y0:Y1, X0:X1]); ax.set_title(f"{r['score']:.2f} | {100 * r['area'] / P:.2f} %", fontsize=6, pad=1)
fig.savefig(a.out, dpi=170); print('[sheet] wrote', a.out)
