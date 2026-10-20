#!/usr/bin/env python3
"""super_fig.py (2026-10-20, CVPR super hierarchy) — the one figure of the decisive test: a held-out frame with the composed query
"trunk of tree k" under the three fields, from super_eval.py's dump: (a) the super field, node path (tree argmax = k and part argmax = k's
trunk node); (b) HiGH as published, word path ("trunk" relevancy >= 0.5 inside tree k's argmax); (c) the LangSplat-style field, "trunk"
relevancy >= 0.5 over the frame (no identity). Colours: true positive green, false positive orange, miss red against the SAM 3 trunk
masks of tree k; the IoU of each panel printed. system python3:  super_fig.py <parts dir> <frame> <tree> <nodes.npz> <out.png> [--crop y0,y1,x0,x1]"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('parts'); ap.add_argument('frame'); ap.add_argument('tree', type=int); ap.add_argument('nodes'); ap.add_argument('out'); ap.add_argument('--crop', default=''); ap.add_argument('--rel', type=float, default=0.5); ap.add_argument('--dump-dir', default='super_dump'); a = ap.parse_args()
P = Path(a.parts); stem = a.frame.replace('.png', ''); d = np.load(P / a.dump_dir / f'{stem}.npz'); H, W = d['tree_super'].shape; k = a.tree; wi = list(d['words']).index('trunk')
nz = np.load(a.nodes); trunk_node = next((int(i) for i, t, ty in zip(nz['ids'], nz['tree'], nz['type']) if int(t) == k and int(ty) == 0), None)
recs = json.load(open(P / 'verdict' / f'{stem}.json')); zp = np.load(P / 'verdict' / f'{stem}.npz'); Hf, Wf = zp['shape']; M = np.unpackbits(zp['masks'], axis=1)[:, :Hf * Wf].reshape(-1, Hf, Wf).astype(bool)
truth = np.zeros((Hf, Wf), bool)
for i, r in enumerate(recs):
    if r['prompt'] == 'tree trunk' and r['tree'] == k: truth |= M[i]
truth = np.array(Image.fromarray(truth).resize((W, H), Image.NEAREST))
rgb = d['rgb'] if d['rgb'].size else np.array(Image.open(P.parent / 'images' / a.frame).convert('RGB').resize((W, H)))
panels = [('super hierarchy, node path: "trunk of tree %d"' % k, (d['tree_super'] == k) & d['live_super'] & (d['part_super'] == trunk_node) if trunk_node is not None else np.zeros((H, W), bool)),
          ('HiGH as published, word path: "trunk" in tree %d' % k, (d['tree_v3'] == k) & d['live_v3'] & (d['rel_v3'][..., wi] >= a.rel)),
          ('LangSplat-style, word path: "trunk" (no identity)', d['live_lite'] & (d['rel_lite'][..., wi] >= a.rel))]
crop = tuple(int(v) for v in a.crop.split(',')) if a.crop else (0, int(H * 0.83), 0, W)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.2 / 3 * (crop[1] - crop[0]) / (crop[3] - crop[2]))); fig.subplots_adjust(0.003, 0.004, 0.997, 0.996, 0.012, 0)
for ax, (label, pred) in zip(axes, panels):
    img = rgb.astype(float) / 255 * 0.5; img[pred & truth] = 0.3 * img[pred & truth] + 0.7 * np.array([0.1, 0.8, 0.2]); img[pred & ~truth] = 0.3 * img[pred & ~truth] + 0.7 * np.array([1.0, 0.55, 0.0]); img[~pred & truth] = 0.3 * img[~pred & truth] + 0.7 * np.array([0.9, 0.1, 0.1])
    u = (pred | truth).sum(); iou = (pred & truth).sum() / u if u else 0.0
    ax.imshow(np.clip(img[crop[0]:crop[1], crop[2]:crop[3]], 0, 1)); ax.axis('off')
    ax.text(0.015, 0.95, label, fontsize=6, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    ax.text(0.985, 0.05, f'IoU {iou:.2f}', fontsize=6.5, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0)); print(f'[fig] {label}: IoU {iou:.3f}, predicted px {int(pred.sum())}, truth px {int(truth.sum())}')
fig.savefig(a.out, dpi=220); print('[fig] wrote', a.out)
