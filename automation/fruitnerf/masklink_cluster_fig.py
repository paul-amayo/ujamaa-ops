#!/usr/bin/env python3
"""masklink_cluster_fig.py <tree> [tag] [dump_tag] [rank] — "show me how this is happening" (Paul, 10-15 16:3x): the view where the rank-th
largest CLUSTER component (members hit >= 3 distinct SAM3 masks in the median view) has the most votes, with every visible fruit gaussian
drawn at its pixel coloured by component (the cluster in red, other counted apples in their own colours, uncounted fragments grey) and the
SAM3 masks outlined by status (green = explained by one counted apple, white = no fruit gaussian). nerf_new pixi env."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'plain30k'; dtag = sys.argv[3] if len(sys.argv) > 3 else tag; arg4 = sys.argv[4] if len(sys.argv) > 4 else '0'; fixed_view = int(arg4[5:]) if arg4.startswith('view:') else None; rank = 0 if fixed_view is not None else int(arg4)
S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); R = S / 'fruit_registry'; v = np.load(R / f'masklink_votes_{tag}.npz'); c = np.load(R / f'masklink_components_{dtag}.npz')
comp, apple, names = c['comp'], c['apple'], list(v['names']); vi, lab, row, uv = v['vi'].astype(np.int64), v['lab'].astype(np.int64), v['row'].astype(np.int64), v['uv']
cv = comp[row]; size = np.bincount(comp)
# clusters: per component, median over views of the number of distinct masks its visible members hit
key = cv * 100000 + vi; order = np.argsort(key); ks = key[order]; ls = lab[order]
starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); ends = np.r_[starts[1:], len(ks)]
nl = {}
for s0, e0 in zip(starts, ends):
    cc = ks[s0] // 100000; nl.setdefault(cc, []).append(len(np.unique(ls[s0:e0])))
mstar = {cc: int(np.median(x)) for cc, x in nl.items()}; clusters = sorted([cc for cc, m in mstar.items() if m >= 3 and apple[cc]], key=lambda cc: -size[cc])
if fixed_view is not None:   # draw a given image: every counted apple in its own colour (the 'cluster' highlight = the biggest counted component in that view)
    nm = f'image_{fixed_view}.png'; view = names.index(nm); K = fixed_view; inv = (vi == view) & apple[cv]; g = np.bincount(cv[inv]).argmax() if inv.any() else -1; sel = cv == g
else:
    if not clusters: print('[cluster-fig] no cluster components'); sys.exit()
    g = clusters[min(rank, len(clusters) - 1)]; sel = cv == g; view = np.bincount(vi[sel]).argmax(); nm = names[view]; K = int(nm[6:-4])
print(f"[cluster-fig] {T} ({dtag}): {len(clusters)} cluster components among the counted apples (sizes {[int(size[x]) for x in clusters[:8]]}, masks per view {[mstar[x] for x in clusters[:8]]}); showing component {g} ({size[g]} gaussians, {mstar.get(g, 0)} masks per view) in {nm}; counted apples visible in this view: {len(set(cv[(vi == view) & apple[cv]].tolist()))}")
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
ents = None
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, e in meta['frame_entries'].items():
        if l2i[int(li)] == K: ents = sorted(e, key=lambda e: e.get('score', 0))
inview = vi == view; per = {}
for l_, cc in zip(lab[inview], cv[inview]):
    if apple[cc]: per.setdefault(int(l_), set()).add(int(cc))
voted = set(lab[inview].tolist()); img = np.array(Image.open(S / 'images' / nm)); H, W = img.shape[:2]
fig, axes = plt.subplots(1, 2, figsize=(22, 8)); 
for ax in axes: ax.imshow(img); ax.set_axis_off()
for j, e in enumerate(ents):
    m = mask_utils.decode(e['rle']); st = 'none' if j not in voted else ('one' if len(per.get(j, ())) == 1 else ('multi' if len(per.get(j, ())) >= 2 else 'frag'))
    for ax in axes: ax.contour(m, levels=[0.5], colors=[{'one': 'lime', 'multi': 'red', 'frag': 'orange', 'none': 'white'}[st]], linewidths=1.2)
others = inview & apple[cv] & (cv != g); rng = np.random.RandomState(0); pal = rng.rand(comp.max() + 1, 3) * 0.7 + 0.3
axes[0].scatter(uv[inview & ~apple[cv], 0], uv[inview & ~apple[cv], 1], s=4, c='0.6', label='uncounted fragments'); axes[0].scatter(uv[others, 0], uv[others, 1], s=6, c=pal[cv[others]], label='other counted apples (one colour each)')
axes[0].scatter(uv[sel & inview, 0], uv[sel & inview, 1], s=14, c='red', edgecolors='k', linewidths=0.3, label=f'largest component here: {size[g]} gaussians, {mstar.get(g, 0)} masks per view')
axes[0].legend(loc='lower left', fontsize=9); axes[0].set_title(f'{T} {nm}: visible fruit gaussians by component', fontsize=11)
# zoom on the cluster
u0, v0 = uv[sel & inview, 0], uv[sel & inview, 1]; cx, cy = np.median(u0), np.median(v0); r = max(120, int(2.5 * max(u0.std(), v0.std())))
axes[1].scatter(uv[others, 0], uv[others, 1], s=16, c=pal[cv[others]]); axes[1].scatter(u0, v0, s=30, c='red', edgecolors='k', linewidths=0.4); axes[1].set_xlim(cx - r, cx + r); axes[1].set_ylim(cy + r, cy - r)
axes[1].set_title(f'zoom: component {g} spans the masks it touches ({mstar[g]} per view) — one count for all of them', fontsize=11)
out = R / f'masklink_cluster_{K}_{dtag}.png'; fig.savefig(out, dpi=70, bbox_inches='tight'); print(f'[cluster-fig] {out}')
