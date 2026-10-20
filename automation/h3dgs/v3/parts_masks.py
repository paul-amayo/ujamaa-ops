#!/usr/bin/env python3
"""parts_masks.py (2026-10-20, CVPR super hierarchy, step 2) — masks of masks inside every supervised tree mask of a frame. SAM 3 image
mode has text and box prompts but no automatic "everything" mode, so the implicit levels come from a prompt list (the Clio task list:
default "tree trunk,branch,leaves") run on the CROP of each tree (its supervision mask's bounding box with a margin, so small parts are
seen at full resolution); a mask is kept when its area is >= --min-area px and >= --inside of it lies inside the parent tree mask.
Levels by area: coarse = >= 5 % of the tree mask, fine = the rest; a fine mask nests under the coarse mask of the same tree that holds
>= --inside of it, else under the tree. Every node has exactly one parent. Fruit stays the finest explicit level (ids >= 10000 in the
supervision maps) and is not re-detected here. Per frame: <out>/<frame>.json (records: pid, tree, prompt, level, parent, score, area,
inside, box) + <frame>.npz (masks packed as bits [m, H, W]) + <frame>_ids.png (uint16: 20000 + pid, smallest covering part wins; 0 = none).
sam3 pixi env:  parts_masks.py <identity dir> <images dir> <out dir> [--frames verdict|train|all] [--prompts a,b,c] [--conf 0.3]
                               [--min-area 150] [--inside 0.9] [--tree-min-frac 0.01] [--margin 0.1] [--max-per-prompt 30]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
ap = argparse.ArgumentParser(); ap.add_argument('identity'); ap.add_argument('images'); ap.add_argument('out'); ap.add_argument('--frames', default='verdict')
ap.add_argument('--prompts', default='tree trunk,branch,leaves'); ap.add_argument('--conf', type=float, default=0.3); ap.add_argument('--min-area', type=int, default=150)
ap.add_argument('--inside', type=float, default=0.9); ap.add_argument('--tree-min-frac', type=float, default=0.01); ap.add_argument('--margin', type=float, default=0.1); ap.add_argument('--max-per-prompt', type=int, default=30)
a = ap.parse_args(); N, IM, OUT = Path(a.identity), Path(a.images), Path(a.out); OUT.mkdir(parents=True, exist_ok=True); PROMPTS = [p.strip() for p in a.prompts.split(',') if p.strip()]
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent
sn = json.load(open(N / 'split_names.json')); fj = json.load(open(N / 'frames.json'))['frames']
frames = fj if a.frames == 'verdict' else (sn['train'] if a.frames == 'train' else sorted(set(sn['train']) | set(fj)))
proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.conf); t0 = time.time(); tot = {p: 0 for p in PROMPTS}; n_trees = 0
for fi, f in enumerate(frames):
    if (OUT / f.replace('.png', '.json')).exists(): continue
    im = Image.open(IM / f).convert('RGB'); W_, H_ = im.size; ids = np.array(Image.open(sup / f)); P = ids.size
    if ids.shape != (H_, W_): ids = np.array(Image.fromarray(ids).resize((W_, H_), Image.NEAREST))
    recs, masks = [], []
    for t in [int(v) for v in np.unique(ids) if v < 10000]:
        tm = ids == t; area_t = int(tm.sum())
        if area_t < a.tree_min_frac * P: continue
        n_trees += 1; ys, xs = np.where(tm); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; my, mx = int(a.margin * (y1 - y0)), int(a.margin * (x1 - x0))
        y0, y1, x0, x1 = max(0, y0 - my), min(H_, y1 + my), max(0, x0 - mx), min(W_, x1 + mx); crop = im.crop((x0, y0, x1, y1))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): st = proc.set_image(crop)
        for prompt in PROMPTS:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): r = proc.set_text_prompt(state=st, prompt=prompt)
            sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0)
            if not len(sc): continue
            ms = r['masks'].cpu().numpy().reshape(len(sc), crop.height, crop.width); kept = 0
            for j in np.argsort(-sc):
                if kept >= a.max_per_prompt: break
                full = np.zeros((H_, W_), bool); full[y0:y1, x0:x1] = ms[j]; area = int(full.sum())
                if area < a.min_area: continue
                inside = float((full & tm).sum() / area)
                if inside < a.inside: continue
                bys, bxs = np.where(full); recs.append({'pid': len(recs) + 1, 'tree': t, 'prompt': prompt, 'score': round(float(sc[j]), 4), 'area': area, 'tree_area': area_t, 'frac_of_tree': round(area / area_t, 4), 'inside': round(inside, 3), 'box': [int(bxs.min()), int(bys.min()), int(bxs.max()) + 1, int(bys.max()) + 1]})
                masks.append(full); kept += 1; tot[prompt] += 1
    # levels and nesting: coarse >= 5 % of the tree; a fine mask nests under the coarse mask of its tree holding >= --inside of it
    for i, r in enumerate(recs):
        r['level'] = 'coarse' if r['frac_of_tree'] >= 0.05 else 'fine'; r['parent'] = f"tree:{r['tree']}"
    for i, r in enumerate(recs):
        if r['level'] != 'fine': continue
        best = None
        for j, q in enumerate(recs):
            if q['level'] == 'coarse' and q['tree'] == r['tree'] and j != i:
                frac = float((masks[i] & masks[j]).sum() / r['area'])
                if frac >= a.inside and (best is None or q['area'] < recs[best]['area']): best = j
        if best is not None: r['parent'] = f"part:{recs[best]['pid']}"
    lab = np.zeros((H_, W_), np.uint16)
    for i in np.argsort([-r['area'] for r in recs]): lab[masks[i]] = 20000 + recs[i]['pid']   # smallest covering part wins
    json.dump(recs, open(OUT / f.replace('.png', '.json'), 'w')); Image.fromarray(lab).save(OUT / f.replace('.png', '_ids.png'))
    np.savez_compressed(OUT / f.replace('.png', '.npz'), masks=np.packbits(np.array(masks, bool).reshape(len(masks), -1), axis=1) if masks else np.zeros((0, 0), np.uint8), shape=np.array([H_, W_]))
    print(f"[parts] {f}: {len(recs)} parts in {len({r['tree'] for r in recs})} trees ({', '.join(f'{p} {sum(1 for r in recs if r[chr(112)+chr(114)+chr(111)+chr(109)+chr(112)+chr(116)] == p)}' for p in PROMPTS)}), coarse {sum(1 for r in recs if r['level'] == 'coarse')}, fine {sum(1 for r in recs if r['level'] == 'fine')}, nested under a part {sum(1 for r in recs if r['parent'].startswith('part'))}; {(time.time() - t0) / (fi + 1):.1f} s/frame", flush=True)
print(f"[parts] done: {len(frames)} frames, {n_trees} tree crops, masks per prompt {tot}, {(time.time() - t0) / 60:.1f} min -> {OUT}")
