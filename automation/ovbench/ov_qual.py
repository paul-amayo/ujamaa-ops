#!/usr/bin/env python3
"""ov_qual.py — qualitative relevancy panel for one method on one frame (ovbench, 2026-11-10).
LERF's own presentation (lerf.py get_outputs_for_camera_ray_bundle 'composited_i'): p = clip(relevancy - 0.5, 0, 1) / max(p),
turbo colormap where relevancy >= 0.5, the method's RGB render elsewhere. White contour = the supervised head the query is
scored against; the title carries IoU at relevancy >= 0.5 and the oracle IoU from scores.jsonl.
  ov_qual.py --rel <dir>/<frame>.npz --scores scores.jsonl --frame kf_002411.png --queries "tree" "the tree on the right" "row 4"
             --sup <dir> --hier <json> --title LERF --out fig.png"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--rel', required=True); ap.add_argument('--scores', required=True); ap.add_argument('--frame', required=True)
ap.add_argument('--queries', nargs='+', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--hier', required=True)
ap.add_argument('--title', default=''); ap.add_argument('--out', required=True)
a = ap.parse_args()
z = np.load(a.rel); S = z['S'].astype(np.float32); Q = [str(q) for q in z['queries']]; rgb = z['rgb'].astype(np.float32) / 255; H, W, _ = S.shape
lines = {l['query']: l for l in map(json.loads, open(a.scores)) if l['frame'] == a.frame}
Hj = json.load(open(a.hier)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tof = {f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
g = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / a.frame), np.uint16)).resize((W, H), Image.NEAREST))
def gt_for(q):
    if q == 'tree': return g < 65535
    if q == 'orange': return (g >= 10000) & (g < 65535)
    l = lines.get(q)
    if l is None: return None
    if l['kind'] in ('TREE', 'RIGHT'): return (g == l['id']) | np.isin(g, [f + 10000 for f, t in tof.items() if t == l['id']])
    if l['kind'] == 'ROW':
        ts = [t for t, r in row_of.items() if r == l['id']]; return np.isin(g, ts) | np.isin(g, [f + 10000 for f, t in tof.items() if t in ts])
cm = plt.get_cmap('turbo')
fig, ax = plt.subplots(1, len(a.queries) + 1, figsize=(4.2 * (len(a.queries) + 1), 4.2 * H / W + 0.6))
ax[0].imshow(rgb); ax[0].set_title(f'{a.title} render, {a.frame[:-4]}', fontsize=10)
for k, q in enumerate(a.queries, 1):
    r = S[..., Q.index(q)]; p = np.clip(r - 0.5, 0, 1); p = p / (p.max() + 1e-6); img = np.where((r >= 0.5)[..., None], cm(p)[..., :3], rgb)
    ax[k].imshow(img); m = gt_for(q)
    if m is not None and m.any(): ax[k].contour(m.astype(float), levels=[0.5], colors='white', linewidths=0.8)
    l = lines.get(q); sub = f"IoU@0.5 {l['rel05']['iou']:.2f} | oracle {l['oracle']['iou']:.2f} | max rel {l['max_rel']:.2f}" if l else 'no supervised head in frame'
    lab = q + (f" (= tree {l['id']})" if l and l['kind'] == 'RIGHT' else '')
    ax[k].set_title(f'"{lab}"\n{sub}', fontsize=9)
for x in ax: x.axis('off')
plt.tight_layout(); plt.savefig(a.out, dpi=150); print(f'[qual] {a.out}')
