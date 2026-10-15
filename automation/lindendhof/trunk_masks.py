#!/usr/bin/env python3
"""Lindendhof identity, step 1 (2026-10-12, design approved by Paul: trees = trunks, canopy split per tree in 3D, no fruit level):
SAM3 image mode, prompt "tree trunk" (the probe: 10/10 frames, ~24 masks/frame at 0.3), on the frames of each lane's v3 workspace that
the census and the verdicts will use: every held-out frame + every STRIDE-th training frame. Per frame: <out>/<lane>/<frame>.png =
uint8 instance index (0 = none; higher score painted last, so it wins overlaps) + <frame>.json [{idx, score, box, area}].
sam3 pixi env:  trunk_masks.py <survey root> <out root> [lanes=1,2,3,4,5] [stride=3] [conf=0.3] [min_area=150]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
S, OUT = Path(sys.argv[1]), Path(sys.argv[2]); LANES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4,5').split(',')]
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 3; CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3; MIN_AREA = int(sys.argv[6]) if len(sys.argv) > 6 else 150
import os; PROMPT = os.environ.get('PROMPT', 'tree trunk')   # PROMPT env (2026-10-12: Klapmuts Dec anchors = 'grow bag')
WSMAP = dict(kv.split(':') for kv in os.environ.get('WSMAP', '').split(',') if kv)   # WSMAP='2:chunk_lane' when a lane's v3 workspace has another name
proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=CONF)
for n in LANES:
    WS = S / 'experimental/v3' / WSMAP.get(str(n), f'chunk_lane{n}'); sp = json.load(open(WS / 'split.json')); frames = sorted(set(sp['test']) | set(sp['train'][::STRIDE]))
    od = OUT / f'lane{n}'; od.mkdir(parents=True, exist_ok=True); t0 = time.time(); tot = 0
    for i, f in enumerate(frames):
        if (od / f.replace('.png', '.json')).exists(): continue
        im = Image.open(WS / 'images' / f).convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            st = proc.set_image(im); r = proc.set_text_prompt(state=st, prompt=PROMPT)
        sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0)
        ms = r['masks'].cpu().numpy().reshape(len(sc), im.height, im.width) if len(sc) else np.zeros((0, im.height, im.width), bool)
        bx = r['boxes'].float().cpu().numpy() if len(sc) else np.zeros((0, 4))
        lab = np.zeros((im.height, im.width), np.uint8); inst = []
        for j in np.argsort(sc):   # ascending: the highest score is painted last and owns shared pixels
            a = int(ms[j].sum())
            if a < MIN_AREA or len(inst) >= 254: continue
            k = len(inst) + 1; lab[ms[j]] = k; inst.append({'idx': k, 'score': round(float(sc[j]), 4), 'box': [round(float(v), 1) for v in bx[j]], 'area': a})
        Image.fromarray(lab).save(od / f); json.dump(inst, open(od / f.replace('.png', '.json'), 'w')); tot += len(inst)
        if i % 50 == 0: print(f'[trunks] lane {n}: {i + 1}/{len(frames)} frames, {tot} masks ({(time.time() - t0) / (i + 1):.2f} s/frame)', flush=True)
    print(f'[trunks] lane {n} done: {len(frames)} frames ({len(sp["test"])} held-out + every {STRIDE}th of {len(sp["train"])} training), {tot} masks, {(time.time() - t0) / 60:.1f} min', flush=True)
