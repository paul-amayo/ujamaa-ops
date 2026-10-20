#!/usr/bin/env python3
"""trunk_allmasks.py (2026-10-20) — every "tree trunk" SAM 3 finds on the whole frame (no tree assignment, no inside rule), for checking the
composed-query reference: a field prediction outside the tree-assigned reference may still sit on a real trunk the reference dropped.
sam3 pixi env:  trunk_allmasks.py <images dir> <out dir> <frame> [<frame> ...] [--conf 0.3] [--min-area 150]"""
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
ap = argparse.ArgumentParser(); ap.add_argument('images'); ap.add_argument('out'); ap.add_argument('frames', nargs='+'); ap.add_argument('--conf', type=float, default=0.3); ap.add_argument('--min-area', type=int, default=150); a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.conf)
for f in a.frames:
    im = Image.open(Path(a.images) / f).convert('RGB')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): st = proc.set_image(im); r = proc.set_text_prompt(state=st, prompt='tree trunk')
    sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0); ms = r['masks'].cpu().numpy().reshape(len(sc), im.height, im.width) if len(sc) else np.zeros((0, im.height, im.width), bool)
    keep = [j for j in range(len(sc)) if ms[j].sum() >= a.min_area]; M = ms[keep].astype(bool)
    np.savez_compressed(OUT / f.replace('.png', '.npz'), masks=np.packbits(M.reshape(len(keep), -1), axis=1) if keep else np.zeros((0, 0), np.uint8), scores=sc[keep], shape=np.array([im.height, im.width]))
    print(f'[trunks] {f}: {len(keep)} masks (scores {np.round(sc[keep], 2).tolist()})')
