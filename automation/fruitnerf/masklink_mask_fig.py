#!/usr/bin/env python3
"""masklink_mask_fig.py <tree> <image_k> [tag] — one training view with every SAM3 apple mask outlined by its registry status: green = explained
by exactly one counted apple, red = by >= 2 counted apples (split or touching pair), orange = only by uncounted fragments, grey = no visible
fruit gaussian. Same status logic as masklink_mask_check.py. nerf_new pixi env."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T, K = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else 'plain30k'; S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); R = S / 'fruit_registry'
v = np.load(R / f'masklink_votes_{tag}.npz'); c = np.load(R / f'masklink_components_{tag}.npz'); comp = c['comp']; apple = c['apple']; names = list(v['names']); nm = f'image_{K}.png'; vi = names.index(nm)
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
ents = None
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, e in meta['frame_entries'].items():
        if l2i[int(li)] == K: ents = sorted(e, key=lambda e: e.get('score', 0))
sel = v['vi'] == vi; labs = v['lab'][sel]; ac = np.where(apple[comp[v['row'][sel]]], comp[v['row'][sel]], -1)
per = defaultdict(set); voted = set(int(l) for l in labs)
for l, a in zip(labs, ac):
    if a >= 0: per[int(l)].add(int(a))
img = np.array(Image.open(S / 'images' / nm)); fig, ax = plt.subplots(figsize=(16, 10.7)); ax.imshow(img); ax.set_axis_off(); cnt = defaultdict(int)
for j, e in enumerate(ents):
    m = mask_utils.decode(e['rle']); s = 'none' if j not in voted else ('one' if len(per.get(j, ())) == 1 else ('multi' if len(per.get(j, ())) >= 2 else 'frag')); cnt[s] += 1
    ax.contour(m, levels=[0.5], colors=[{'one': 'lime', 'multi': 'red', 'frag': 'orange', 'none': 'white'}[s]], linewidths=1.6)
ax.set_title(f"{T} {nm} ({tag}): SAM3 apple masks {len(ents)} — one counted apple {cnt['one']} (green), >= 2 apples {cnt['multi']} (red), uncounted fragments only {cnt['frag']} (orange), no visible fruit gaussian {cnt['none']} (white)", fontsize=11)
out = R / f'masklink_masks_{K}_{tag}.png'; fig.savefig(out, dpi=80, bbox_inches='tight'); print(f'[mask-fig] {out}: ' + ', '.join(f'{k} {n}' for k, n in cnt.items()))
