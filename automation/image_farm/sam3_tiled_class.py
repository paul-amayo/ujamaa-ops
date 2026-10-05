#!/usr/bin/env python3
"""Class-level SAM3 masks from overlapping TILES (UJAMAA 2026-10-05; Paul: "why can't we train with those, pepper masks" -> "go
class-level pepper"). The full-frame probe (build_tree_instances.py, prompt 'pepper') found pods on only 50 of 241 chilli frames:
SAM3 resizes the whole 1080x1920 frame to its ~1008 px input (pods shrink to a few px), the image-level presence score multiplies
every mask, and the probe's --min-mask-area 400 drops small pods. Here each frame is cut into a grid of overlapping tiles, each
tile is its own SAM3 image (own presence score, ~2x the frame's resolution at 3x4), masks >= --conf and >= --min-area pixels
(frame pixels) are OR-ed into one class map per frame. Class level: no instance ids, no tracking.
sam3 pixi env:  python sam3_tiled_class.py --images <dir of image_N.png> --frames 0-141 --prompt pepper --out <dir>
-> <out>/class/image_N.png (uint8, 255 = prompt class), <out>/stats.jsonl (per frame: masks, class px, per-tile counts)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/sam3')
ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--frames', default='')
ap.add_argument('--prompt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--grid', default='3x4', help='cols x rows')
ap.add_argument('--overlap', type=float, default=0.15); ap.add_argument('--conf', type=float, default=0.3); ap.add_argument('--min-area', type=int, default=20)
a = ap.parse_args(); OUT = Path(a.out); (OUT / 'class').mkdir(parents=True, exist_ok=True)
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
model = build_sam3_image_model(); proc = Sam3Processor(model, confidence_threshold=a.conf)
imgs = sorted(Path(a.images).glob('image_*.png'), key=lambda p: int(p.stem.split('_')[1]))
if a.frames:
    lo, hi = (int(x) for x in a.frames.split('-')); imgs = [p for p in imgs if lo <= int(p.stem.split('_')[1]) <= hi]
cols, rows = (int(x) for x in a.grid.split('x')); t0 = time.time(); stats = open(OUT / 'stats.jsonl', 'a')
for k, p in enumerate(imgs):
    if (OUT / 'class' / p.name).exists(): continue
    im = Image.open(p).convert('RGB'); W, H = im.size; cls = np.zeros((H, W), bool); n_m, per_tile = 0, []
    tw, th = int(W / cols * (1 + a.overlap)), int(H / rows * (1 + a.overlap))
    xs = np.linspace(0, W - tw, cols).astype(int); ys = np.linspace(0, H - th, rows).astype(int)
    for y0 in ys:
        for x0 in xs:
            crop = im.crop((x0, y0, x0 + tw, y0 + th))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                st = proc.set_image(crop); out = proc.set_text_prompt(state=st, prompt=a.prompt)
            masks = out.get('masks'); c = 0
            if masks is not None and len(masks):
                for m in masks:
                    arr = (m.detach().float().cpu().numpy() if hasattr(m, 'detach') else np.asarray(m)).squeeze() > 0.5
                    if arr.shape != (th, tw) or arr.sum() < a.min_area: continue
                    cls[y0:y0 + th, x0:x0 + tw] |= arr; c += 1
            per_tile.append(c); n_m += c
    Image.fromarray((cls * 255).astype(np.uint8)).save(OUT / 'class' / p.name)
    stats.write(json.dumps({'frame': p.name, 'masks': n_m, 'class_px': int(cls.sum()), 'tiles': per_tile}) + '\n'); stats.flush()
    if k % 10 == 0: print(f'[tiled] {k + 1}/{len(imgs)} {p.name}: {n_m} masks, {int(cls.sum())} px ({time.time() - t0:.0f}s)', flush=True)
print(f'[tiled] done {len(imgs)} frames in {time.time() - t0:.0f}s -> {OUT}', flush=True)
