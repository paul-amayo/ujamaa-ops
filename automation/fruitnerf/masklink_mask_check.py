#!/usr/bin/env python3
"""masklink_mask_check.py <tree> [tag] — GT-free quality check of the mask-linked registry: for every SAM3 apple mask in every view, how many
counted apples (components) have a visible fruit gaussian inside it. Exactly one = the mask is explained by one apple; >= 2 = a split apple
(or touching apples under one mask); 0 = the mask has no visible fruit gaussian (missed by the splat / census). Reads the votes cache and the
component dump written by fruit_registry_masklink.py (needs comp in the dump). nerf_new pixi env."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
T = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'plain30k'; dtag = sys.argv[3] if len(sys.argv) > 3 else tag; S = Path(T) if '/' in T else Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); T = S.name; R = S / 'fruit_registry'
v = np.load(R / f'masklink_votes_{tag}.npz'); c = np.load(R / f'masklink_components_{dtag}.npz'); comp = c['comp']; apple = c['apple']; names = list(v['names'])
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
from pycocotools import mask as mask_utils
nmask = {}; area = {}   # mask areas (px) in the labmap order of fruit_registry_masklink.py: sorted by score ascending
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items(): nm = f"image_{l2i[int(li)]}.png"; nmask[nm] = len(ents); area[nm] = [float(mask_utils.area(e['rle'])) for e in sorted(ents, key=lambda e: e.get('score', 0))]
total = sum(nmask.get(n, 0) for n in names)
key = v['vi'].astype(np.int64) * 100000 + v['lab'].astype(np.int64); ac = np.where(apple[comp[v['row']]], comp[v['row']], -1)   # apple component per vote (-1 = not a counted apple)
per = defaultdict(set); anyvote = set()
for k, a in zip(key, ac):
    anyvote.add(k)
    if a >= 0: per[k].add(int(a))
# dominant-share view of the same masks: votes per (mask, counted apple); explained = the top apple holds >= 80 % of the mask's counted votes, contested = the runner-up holds >= 20 %
from collections import Counter
vc = Counter(zip(key[ac >= 0].tolist(), ac[ac >= 0].tolist())); top = {}; 
for (k, a_), n_ in vc.items(): top.setdefault(k, []).append(n_)
dom = np.array([sorted(x, reverse=True)[0] / sum(x) for x in top.values()]); run = np.array([(sorted(x, reverse=True)[1] / sum(x)) if len(x) > 1 else 0.0 for x in top.values()])
n_apples = np.array([len(s) for s in per.values()]); hist = np.bincount(n_apples, minlength=4)
masks_voted = len(anyvote); masks_one = int((n_apples == 1).sum()); masks_multi = int((n_apples >= 2).sum()); masks_only_uncounted = masks_voted - len(per)
print(f"[mask-check] {T} ({tag}): {total} SAM3 apple masks over {len(names)} views; with a visible fruit gaussian {masks_voted} ({100*masks_voted/total:.0f} %); "
      f"explained by exactly ONE counted apple {masks_one} ({100*masks_one/total:.0f} %), by >= 2 apples {masks_multi} ({100*masks_multi/total:.0f} %; of which 2: {int((n_apples==2).sum())}, 3+: {int((n_apples>=3).sum())}), "
      f"only by uncounted fragments {masks_only_uncounted} ({100*masks_only_uncounted/total:.0f} %), no fruit gaussian at all {total-masks_voted} ({100*(total-masks_voted)/total:.0f} %)")
print(f"[mask-check] dominant-share: of the {len(dom)} masks with counted votes, top apple >= 80 % of the votes in {100 * np.mean(dom >= 0.8):.0f} %, runner-up >= 20 % (contested) in {100 * np.mean(run >= 0.2):.0f} %")
if 'masks' in v.files and len(v['masks']):   # missed masks inside the tree: lifted centroids of the masks with no vote, against the ring and the row cuts of the dump
    mk = v['masks']; LOOK, UP, cam_r = c['LOOK'], c['UP'], float(c['cam_r']); e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
    X = np.stack([(mk[:, 2:5] - LOOK) @ e1, (mk[:, 2:5] - LOOK) @ e2], 1) / cam_r; inring = np.linalg.norm(X, axis=1) <= 1.0; mkey = mk[:, 0].astype(np.int64) * 100000 + mk[:, 1].astype(np.int64)
    st_none = np.array([k not in anyvote for k in mkey]); st_one = np.array([len(per.get(k, ())) == 1 for k in mkey]); nv_ = len(names)
    print(f"[mask-check] lifted masks: inside the camera ring {100 * inring.mean():.0f} % of all masks; of the masks with NO fruit gaussian {100 * inring[st_none].mean():.0f} % are inside the ring = {inring[st_none].sum() / nv_:.1f} per view (explained-by-one masks inside: {inring[st_one].sum() / nv_:.1f} per view) -> missed share inside the tree {100 * inring[st_none].sum() / max(1, inring[st_none].sum() + inring[st_one].sum()):.0f} %")
# areas by status: are the unexplained masks the small (background / neighbouring-tree) ones?
st = {}
for vi, nm in enumerate(names):
    for j, ar in enumerate(area.get(nm, [])):
        k = vi * 100000 + j; st.setdefault('none' if k not in anyvote else ('one' if len(per.get(k, ())) == 1 else ('multi' if len(per.get(k, ())) >= 2 else 'frag')), []).append(ar)
small = {k: int(np.sum(np.array(v) <= 1024)) for k, v in st.items()}; large = {k: len(v) - small[k] for k, v in st.items()}; ns, nl = sum(small.values()), sum(large.values())
print(f"[mask-check] by size (COCO-small cutoff 1024 px): small masks {ns}: one apple {100 * small.get('one', 0) / max(ns, 1):.0f} %, >= 2 {100 * small.get('multi', 0) / max(ns, 1):.0f} %, none {100 * small.get('none', 0) / max(ns, 1):.0f} %; large masks {nl}: one apple {100 * large.get('one', 0) / max(nl, 1):.0f} %, >= 2 {100 * large.get('multi', 0) / max(nl, 1):.0f} %, none {100 * large.get('none', 0) / max(nl, 1):.0f} %")
print('[mask-check] mask area px by status: ' + '; '.join(f"{k} n={len(v)} median {np.median(v):.0f} (<= 500 px: {100*np.mean(np.array(v) <= 500):.0f} %)" for k, v in st.items()))
# per-apple: in how many of its views does it share a mask with another counted apple
share = defaultdict(int); seen = defaultdict(int)
for s in per.values():
    for a in s: seen[a] += 1; share[a] += (len(s) >= 2)
fr = np.array([share[a] / seen[a] for a in seen]); print(f"[mask-check] counted apples {len(seen)}: share a mask with another counted apple in a median {100*np.median(fr):.0f} % of their views; {int((fr > 0.5).sum())} apples do so in > 50 % of views (likely over-splits or touching pairs)")
