#!/usr/bin/env python3
"""ladder_fig.py (2026-10-17) — the HiGH paper's qualitative containment figure: four address queries on one held-out frame of citrus 05 cell 1_0
(two trees, two rows), each read by containment at one threshold (0.85, the hero figure's) and compared with the supervision id map: true
positives green, false positives orange, misses red, with the IoU printed. Data: hero_heat.py's dump (every address's containment heat) and the
supervision id map. The queries are chosen by the supervision: the two largest trees in the frame, the row of the largest and the other row
present (or, if the frame holds one row, a row of the cell that is absent, where nothing should light).
system python3:  ladder_fig.py <figures dir> [--thr 0.85] [--out fig_containment_qual.png]"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('figdir'); ap.add_argument('--thr', type=float, default=0.85); ap.add_argument('--frame', default='kf_002411.png'); ap.add_argument('--out', default='fig_containment_qual.png'); a = ap.parse_args()
FIG = Path(a.figdir); z = np.load(FIG / 'hero_heat_05_1_0_kf002411.npz', allow_pickle=True); meta = json.loads(str(z['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
H = json.load(open('/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json')); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}
SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
rgb = z['rgb']; Hh, Ww = rgb.shape[:2]; ids = np.array(Image.fromarray(np.array(Image.open(SUP / a.frame))).resize((Ww, Hh), Image.NEAREST))
trees = sorted(((int((ids == t).sum()), int(t)) for t in np.unique(ids) if t < 10000), reverse=True); top = [t for n, t in trees[:2]]
rows_px = {}
for n, t in trees:
    r = row_of.get(t, -1)
    if r >= 0: rows_px[r] = rows_px.get(r, 0) + n
rows_sorted = sorted(rows_px, key=lambda r: -rows_px[r]); r1 = row_of[top[0]]; others = [r for r in rows_sorted if r != r1 and rows_px[r] > 200]
r2 = others[0] if others else next(r for r in sorted(rw) if r not in rows_px)
def truth_of(kind, k): return (ids == k) if kind == 'tree' else np.isin(ids, [t for t, r in row_of.items() if r == k])
queries = [('tree', top[0]), ('tree', top[1]), ('row', r1), ('row', r2)]
fig, axes = plt.subplots(2, 2, figsize=(5.2, 2.5)); fig.subplots_adjust(0.004, 0.004, 0.996, 0.996, 0.015, 0.03)
for ax, (kind, k) in zip(axes.ravel(), queries):
    heat = z[wt[k]] if kind == 'tree' else z[rw[k]]; pred = heat >= a.thr; truth = truth_of(kind, k); u = (pred | truth).sum(); iou = float((pred & truth).sum() / u) if u else float('nan')
    img = rgb.astype(float) / 255.0 * 0.5; img[pred & truth] = 0.3 * img[pred & truth] + 0.7 * np.array([0.1, 0.8, 0.2]); img[pred & ~truth] = 0.3 * img[pred & ~truth] + 0.7 * np.array([1.0, 0.55, 0.0]); img[~pred & truth] = 0.3 * img[~pred & truth] + 0.7 * np.array([0.9, 0.1, 0.1])
    ax.imshow(np.clip(img[:300], 0, 1)); ax.axis('off'); ax.text(0.015, 0.95, f'query: "{kind} {k}"', fontsize=12, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=3, lw=0))
    corner = f'IoU {iou:.2f}' if u and truth.any() else ('absent row: nothing lit' if not pred.any() else f'absent row: {int(pred.sum())} px lit'); ax.text(0.985, 0.05, corner, fontsize=11, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=3, lw=0))
    print(f'[ladder] {kind} {k}: truth {int(truth.sum())} px, lit {int(pred.sum())} px, IoU {iou:.3f}, precision {float((pred & truth).sum() / max(1, pred.sum())):.3f}, recall {float((pred & truth).sum() / max(1, truth.sum())):.3f}')
fig.savefig(FIG / a.out, dpi=150); print('[ladder] wrote', FIG / a.out)
