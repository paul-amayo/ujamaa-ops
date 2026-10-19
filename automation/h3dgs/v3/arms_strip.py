#!/usr/bin/env python3
"""arms_strip.py (2026-10-19) — qualitative strips of the HiGH paper's ablations on one held-out frame of one cell, each panel the paper's
own readout: E2 (geometry arms) the row asked by its address under the argmax rule, E2b (seeding arms) a tree under the argmax rule, E3
(category) "tree" at a relevancy of 0.5 for the hyperbolic, flat and LangSplat-style fields. Inputs are argmax_eval.py dumps (--dump-pred
for the identity arms, --dump-cat for the category fields) under <identity>/dumps/<arm>/<frame>.npz, the lite relevancy npz for E3, and
the cell's supervision; colouring TP green / FP orange / miss red with the IoU printed.
system python3:  arms_strip.py --identity <dir> --frame kf_002411.png --paper <dir> --e2 "A=hyper,D=hyper_own,C=flat96,B=euclid" --row 4
                              --e2b "seed=v3,learned=zero3000,refined=refine3000,projection=projvote" --tree 36 --e3-lite <lite npz>"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--frame', required=True); ap.add_argument('--paper', required=True)
ap.add_argument('--e2', default=''); ap.add_argument('--row', type=int, default=4); ap.add_argument('--e2b', default=''); ap.add_argument('--tree', type=int, default=36)
ap.add_argument('--e3', default=''); ap.add_argument('--e3-lite', default=''); ap.add_argument('--rgb', default='', help='npz with the frame rgb (a hero_heat dump); default <paper>/figures/hero_heat_05_1_0_kf002411.npz'); a = ap.parse_args()
N = Path(a.identity); FIG = Path(a.paper) / 'figures'; stem = a.frame.replace('.png', '')
rgb = np.load(a.rgb or (FIG / 'hero_heat_05_1_0_kf002411.npz'), allow_pickle=True)['rgb']; H, W = rgb.shape[:2]
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent; hj = json.load(open(meta['hierarchy_json']))
ids = np.array(Image.fromarray(np.array(Image.open(sup / a.frame))).resize((W, H), Image.NEAREST)); row_of = {int(o['id']): int(o['row_id']) for o in hj['objects']}
def paint(pred, truth):
    img = rgb.astype(float) / 255 * 0.5; img[pred & truth] = 0.3 * img[pred & truth] + 0.7 * np.array([0.1, 0.8, 0.2]); img[pred & ~truth] = 0.3 * img[pred & ~truth] + 0.7 * np.array([1.0, 0.55, 0.0]); img[~pred & truth] = 0.3 * img[~pred & truth] + 0.7 * np.array([0.9, 0.1, 0.1])
    u = (pred | truth).sum(); return np.clip(img[:int(H * 0.83)], 0, 1), float((pred & truth).sum() / u) if u else 0.0
def strip(panels, out, title_fmt):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5, 3.5 / len(panels) * 0.83 * 300 / 640 * 640 / 300 * 0.5 + 0.0) if False else (3.5, 0.75 * 3.5 / len(panels) * (0.83 * H / W) * len(panels) / 1.0 / 1.0)) if False else plt.subplots(1, len(panels), figsize=(3.5, 3.5 / len(panels) * 0.83 * H / W))
    fig.subplots_adjust(0.003, 0.004, 0.997, 0.996, 0.012, 0)
    for ax, (label, pred, truth) in zip(np.atleast_1d(axes), panels):
        img, iou = paint(pred, truth); ax.imshow(img); ax.axis('off')
        ax.text(0.02, 0.95, label, fontsize=5.5, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.2, lw=0))
        ax.text(0.98, 0.05, f'IoU {iou:.2f}', fontsize=5.5, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.2, lw=0)); print(f'[strip] {out.name} {label}: IoU {iou:.3f}')
    fig.savefig(out, dpi=220); plt.close(fig); print('[strip] wrote', out)
if a.e2:
    truth = np.isin(ids, [t for t, r in row_of.items() if r == a.row]); panels = []
    for item in a.e2.split(','):
        name, arm = item.split('='); d = np.load(N / 'dumps' / arm / f'{stem}.npz'); panels.append((name, (d['row'] == a.row) & d['live'], truth))
    strip(panels, FIG / 'fig_e2_strip.png', '')
if a.e2b:
    truth = ids == a.tree; panels = []
    for item in a.e2b.split(','):
        name, arm = item.split('='); d = np.load(N / 'dumps' / arm / f'{stem}.npz'); panels.append((name, (d['tree'] == a.tree) & d['live'], truth))
    strip(panels, FIG / 'fig_e2b_strip.png', '')
if a.e3:
    truth = (ids != 65535) & (ids < 10000); panels = []
    for item in a.e3.split(','):
        name, arm = item.split('='); d = np.load(N / 'dumps' / arm / f'{stem}.npz'); j = list(d['words']).index('tree'); panels.append((name, (d['S'][..., j] >= 0.5) & d['live'], truth))
    if a.e3_lite:
        z = np.load(a.e3_lite, allow_pickle=True); panels.append(('LangSplat-style', (z['tree'] >= 0.5) & ~z['void'].astype(bool), truth))
    strip(panels, FIG / 'fig_e3_strip.png', '')
