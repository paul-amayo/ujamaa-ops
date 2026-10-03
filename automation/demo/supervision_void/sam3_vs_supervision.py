#!/usr/bin/env python3
"""15 sam3_vs_supervision.py (nerf_new env): SAM3's raw masks on a keyframe beside the supervision built from them (UJAMAA
2026-10-03; Paul: "show sam3 masks on 3112"). Top: every SAM3 detection, one colour each, drawn largest first so smaller
masks sit on top (the painter's smallest-on-top rule), labelled with the tree id the lifting gave it or "no id". Bottom: the
chunk supervision the identity training sees (tree ids coloured, unlabelled dark), with SAM3's id-less masks dashed red.
Usage: sam3_vs_supervision.py [K] (default 3112)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pycocotools import mask as mu
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3112
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; NB = S / 'experimental/h3dgs_native/chunk_3_1_sam3'; W, H = 1280, 720
m2g = json.load(open(G / 'global_ids.json'))['member_to_global']
row_of = {o['id']: o['row_id'] for o in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
ents = json.load(open(G / f'clip_{K // 200:03d}/frame_entries.json'))['frame_entries'].get(str(K % 200), [])
dec = sorted([(int(e['obj_id']), mu.decode(e['rle']).astype(bool), int(e['area']), float(e['score'])) for e in ents], key=lambda d: -d[2])
ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{K:06d}.png').convert('RGB')).astype(np.float32)
PAL = [(42, 120, 214), (27, 175, 122), (237, 161, 0), (232, 123, 164), (74, 58, 167), (0, 160, 190), (150, 110, 60), (120, 190, 40), (230, 90, 40), (90, 90, 200), (200, 60, 160), (60, 140, 60), (180, 140, 20), (20, 110, 140)]
fig, axs = plt.subplots(2, 1, figsize=(13, 15), dpi=100)
ax = axs[0]; base = ph * 0.55
for i, (oid, m, area, sc) in enumerate(dec): base[m] = base[m] * 0.35 + np.array(PAL[i % len(PAL)], np.float32) * 0.65
ax.imshow(base.astype(np.uint8)); lab = []
for i, (oid, m, area, sc) in enumerate(dec):
    g = m2g.get(f'{K // 200}:{oid}')
    ax.contour(m, levels=[0.5], colors=['red' if g is None else [c / 255 for c in PAL[i % len(PAL)]]], linewidths=1.6, linestyles='--' if g is None else '-')
    yy, xx = np.nonzero(m); lab.append((float(np.median(xx)), float(np.median(yy)), f'SAM3 #{oid}\n' + (f'-> tree {g} (row {row_of.get(g)})' if g is not None else '-> no id') + f'\n{area / 1000:.1f}k px', g))
lab.sort(); xs = np.linspace(60, W - 60, len(lab))
for i, (x0, y0, t, g) in enumerate(lab):
    ax.annotate(t, xy=(x0, y0), xytext=(xs[i], 45 if i % 2 == 0 else 150), fontsize=7.5, color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#c62828' if g is None else '#222222', alpha=0.9, lw=0), arrowprops=dict(arrowstyle='-', color='white', lw=0.8))
ax.set_axis_off(); ax.set_title(f'kf_{K:06d}: all {len(dec)} SAM3 "tree" detections, one colour each (dashed red outline = the lifting gave it no tree id)', fontsize=11, loc='left')
ax = axs[1]; sup = np.array(Image.open(NB / f'supervision/trees_only/kf_{K:06d}.png'), np.uint16); sv = ph * 0.3
ids = sorted(int(u) for u in np.unique(sup) if u < 10000)
for i, t in enumerate(ids):
    c = np.array(PAL[(i * 3) % len(PAL)], np.float32); sv[sup == t] = sv[sup == t] * 0.3 + c * 0.7
    yy, xx = np.nonzero(sup == t); ax.text(float(np.median(xx)), float(np.median(yy)), f'tree {t}\nrow {row_of.get(t)}', fontsize=9, color='white', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='#222222', alpha=0.8, lw=0))
ax.imshow(sv.astype(np.uint8))
for oid, m, area, sc in dec:
    if m2g.get(f'{K // 200}:{oid}') is None and area >= 3000: ax.contour(m, levels=[0.5], colors=['red'], linewidths=1.6, linestyles='--')
ax.set_axis_off(); ax.set_title(f'kf_{K:06d}: the supervision built from those masks (what training sees): {len(ids)} trees labelled, everything else unlabelled; dashed red = SAM3 masks with no id', fontsize=11, loc='left')
fig.tight_layout(); fn = f'/home/paperspace/data/demo_video_v2/citrus_a_bar/kf{K:06d}_sam3_vs_supervision.png'; fig.savefig(fn); print('[fig]', fn)
