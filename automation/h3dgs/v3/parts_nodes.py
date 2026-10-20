#!/usr/bin/env python3
"""parts_nodes.py (2026-10-20, CVPR super hierarchy, step 3) — the implicit level as Clio-style task nodes: one node per (tree, part type)
over the training views, from the masks parts_masks.py found. Each node's anchor is the mean RAW two-level mask-crop CLIP embedding (the
HiGH MaskCropEncoder, the text bank's CLIP model) of all its masks over the views, so it lives in the space of the address anchors; node
id = 20000 + 4 * tree + type (type: 0 tree trunk, 1 branch, 2 leaves). Also writes the combined supervision maps for the census:
the cell's tree / fruit identity map with part pixels overwritten by their node id (leaves, then branch, then trunk, so the trunk wins;
only where the pixel carried that tree's id; fruit ids stay on top as the finest level).
Outputs under <out>: nodes.npz (ids, emb [n, 512], tree, type, n_masks, n_frames), nodes.json, supervision/<frame>.png + manifest.json.
nerf_new pixi env:  parts_nodes.py <identity dir> <images dir> <parts dir (train)> <out dir> [--min-masks 3]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
import sys; sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from clip_image_embedder import MaskCropEncoder, MaskCropEncoderConfig
ap = argparse.ArgumentParser(); ap.add_argument('identity'); ap.add_argument('images'); ap.add_argument('parts'); ap.add_argument('out'); ap.add_argument('--min-masks', type=int, default=3); a = ap.parse_args()
N, IM, PD, OUT = Path(a.identity), Path(a.images), Path(a.parts), Path(a.out); (OUT / 'supervision').mkdir(parents=True, exist_ok=True); dev = 'cuda'; torch.set_grad_enabled(False)
TYPE = {'tree trunk': 0, 'branch': 1, 'leaves': 2}; ORDER = ['leaves', 'branch', 'tree trunk']   # painted in this order: the trunk wins
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent
enc = MaskCropEncoder(MaskCropEncoderConfig(), device=dev)
def embed(img, masks):
    crops = torch.stack([enc._get_multi_level_crops(torch.from_numpy(m.astype(np.uint8)), img, num_levels=2, pad=True) for m in masks]); out = []
    for j in range(0, len(crops), 16): out.append(enc.model.encode_image(crops[j:j + 16].flatten(0, 1).to(dev), normalize=False).float().view(-1, 2, 512).mean(1).cpu())
    return torch.cat(out).numpy()
acc = {}; t0 = time.time(); n_frames = 0; n_parts = 0
for jf in sorted(PD.glob('kf_*.json')):
    recs = json.load(open(jf)); f = jf.name.replace('.json', '.png'); ids = np.array(Image.open(sup / f)); lab = ids.copy(); n_frames += 1
    if recs:
        z = np.load(PD / jf.name.replace('.json', '.npz')); H, W = z['shape']; M = np.unpackbits(z['masks'], axis=1)[:, :H * W].reshape(-1, H, W).astype(bool)
        img = torch.from_numpy(np.array(Image.open(IM / f).convert('RGB'))).permute(2, 0, 1); E = embed(img, M); n_parts += len(recs)
        for i, r in enumerate(recs):
            k = (r['tree'], TYPE[r['prompt']]); d = acc.setdefault(k, {'sum': np.zeros(512, np.float64), 'n': 0, 'frames': set(), 'area': 0}); d['sum'] += E[i]; d['n'] += 1; d['frames'].add(f); d['area'] += r['area']
        for p in ORDER:
            for i, r in enumerate(recs):
                if r['prompt'] == p: lab[M[i] & (ids == r['tree'])] = 20000 + 4 * r['tree'] + TYPE[p]
        lab[(ids >= 10000) & (ids < 65535)] = ids[(ids >= 10000) & (ids < 65535)]   # fruit stays the finest level
    Image.fromarray(lab.astype(np.uint16), mode='I;16').save(OUT / 'supervision' / f)
keys = sorted(k for k, d in acc.items() if d['n'] >= a.min_masks); dropped = [k for k in acc if k not in keys]
ids_ = np.array([20000 + 4 * t + ty for t, ty in keys]); emb = np.stack([acc[k]['sum'] / acc[k]['n'] for k in keys]).astype(np.float32)
np.savez(OUT / 'nodes.npz', ids=ids_, emb=emb, tree=np.array([k[0] for k in keys]), type=np.array([k[1] for k in keys]), n_masks=np.array([acc[k]['n'] for k in keys]), n_frames=np.array([len(acc[k]['frames']) for k in keys]))
json.dump({'nodes': [{'id': int(i), 'tree': int(k[0]), 'type': int(k[1]), 'type_name': [n for n, v in TYPE.items() if v == k[1]][0], 'n_masks': int(acc[k]['n']), 'n_frames': len(acc[k]['frames']), 'mean_area': int(acc[k]['area'] / acc[k]['n'])} for i, k in zip(ids_, keys)],
           'dropped_below_min_masks': [{'tree': int(t), 'type': int(ty), 'n_masks': int(acc[(t, ty)]['n'])} for t, ty in dropped], 'min_masks': a.min_masks, 'frames': n_frames, 'parts': n_parts, 'anchor': 'mean raw two-level mask-crop CLIP (ViT-B-16 laion2b_s34b_b88k) over the training views'}, open(OUT / 'nodes.json', 'w'), indent=1)
json.dump({'schema': 'supervision_manifest/v1', 'unlabelled_value': 65535, 'fruit_id_base': 10000, 'part_id_base': 20000, 'part_types': TYPE, 'source': str(sup), 'parts': str(PD), 'counts': {'frames': n_frames, 'parts': n_parts, 'nodes': len(keys)}}, open(OUT / 'supervision' / 'manifest.json', 'w'), indent=1)
by_type = {n: sum(1 for k in keys if k[1] == v) for n, v in TYPE.items()}
print(f"[nodes] {n_frames} frames, {n_parts} part masks -> {len(keys)} nodes over {len({k[0] for k in keys})} trees ({by_type}); {len(dropped)} (tree, type) pairs below {a.min_masks} masks dropped; masks per node median {np.median([acc[k]['n'] for k in keys]):.0f}; {time.time() - t0:.0f}s -> {OUT}")
