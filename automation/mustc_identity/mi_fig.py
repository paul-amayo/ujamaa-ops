#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3): one figure, a view with the address masks over the image. Left: every non-void pixel coloured
by the plant whose address wins the plant-level argmax; right: by the row whose address wins the row-level argmax. White outlines =
the supervision masks of that view (held out of the seeding for nikon_5 / nikon_15), labelled with the plant id.
nerf_new pixi env:  mi_fig.py --dir <stop dir> --view nikon_5 --out fig.png [--tag tangent_vf]"""
import argparse, json
import cv2, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tag', default='tangent_vf'); a = ap.parse_args(); D = a.dir
C = json.load(open(f'{D}/cameras.json')); name = next(n for n, c in C['frames'].items() if c['label'] == a.view)
im = cv2.cvtColor(cv2.imread(f'{D}/views/{a.view}.png'), cv2.COLOR_BGR2RGB); Hh, Ww = im.shape[:2]
pr = np.load(f'{D}/pred_{a.tag}/{name}.npz'); gt = cv2.imread(f'{D}/cell/supervision/trees_only/{name}', cv2.IMREAD_UNCHANGED)
cmap = plt.get_cmap('tab20'); fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), dpi=150)
for ax, key, title in ((axs[0], 'tree', 'plant addresses'), (axs[1], 'row', 'row addresses')):
    lab = cv2.resize(pr[key].astype(np.int32), (Ww, Hh), interpolation=cv2.INTER_NEAREST); ov = im.astype(np.float32).copy(); ids = [int(i) for i in np.unique(lab) if i >= 0]
    for k, i in enumerate(ids):
        col = np.array(cmap(k % 20)[:3]) * 255; m = lab == i; ov[m] = 0.45 * ov[m] + 0.55 * col
    ax.imshow(ov.astype(np.uint8)); ax.set_axis_off(); ax.set_title(f'{a.view} ({C["frames"][name]["split"]}): {title}, argmax among {len(ids)} lit', fontsize=10)
    for i in [int(u) for u in np.unique(gt) if u != 65535]:
        m = (gt == i).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs: ax.plot(c[:, 0, 0], c[:, 0, 1], color='white', lw=0.9)
        ys, xs = np.nonzero(m); ax.text(xs.mean(), ys.min() - 8, str(i), color='white', fontsize=7, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5, pad=0.5, lw=0))
fig.tight_layout(); fig.savefig(a.out, bbox_inches='tight'); print('[fig]', a.out)
