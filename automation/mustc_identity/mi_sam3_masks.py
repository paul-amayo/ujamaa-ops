#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3, 2026-11-11), step 2: SAM 3 instance masks on the dumped views.
The image-mode recipe of automation/lindendhof/trunk_masks.py (Sam3Processor, one text prompt, confidence threshold, minimum area;
instances written in ascending score so the highest score owns shared pixels), unchanged except for the input (a folder of views)
and the index image being uint16 (a rig view can hold more than 254 instances). Per view: <out>/<label>.png = instance index
(0 = none) + <label>.json [{idx, score, box, area}].
sam3 pixi env:  PROMPT='plant' mi_sam3_masks.py <views dir> <out dir> [conf=0.3] [min_area=150] [labels=all]"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
VIEWS, OUT = Path(sys.argv[1]), Path(sys.argv[2]); CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3; MIN_AREA = int(sys.argv[4]) if len(sys.argv) > 4 else 150
ONLY = set(sys.argv[5].split(',')) if len(sys.argv) > 5 and sys.argv[5] != 'all' else None
PROMPT = os.environ.get('PROMPT', 'plant'); OUT.mkdir(parents=True, exist_ok=True)
proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=CONF); t0 = time.time(); tot = 0
frames = sorted(p for p in VIEWS.glob('*.png') if ONLY is None or p.stem in ONLY)
for i, f in enumerate(frames):
    im = Image.open(f).convert('RGB')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        st = proc.set_image(im); r = proc.set_text_prompt(state=st, prompt=PROMPT)
    sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0)
    ms = r['masks'].cpu().numpy().reshape(len(sc), im.height, im.width) if len(sc) else np.zeros((0, im.height, im.width), bool)
    bx = r['boxes'].float().cpu().numpy() if len(sc) else np.zeros((0, 4))
    lab = np.zeros((im.height, im.width), np.uint16); inst = []
    for j in np.argsort(sc):   # ascending: the highest score is written last and owns shared pixels
        a = int(ms[j].sum())
        if a < MIN_AREA: continue
        k = len(inst) + 1; lab[ms[j]] = k; inst.append({'idx': k, 'score': round(float(sc[j]), 4), 'box': [round(float(v), 1) for v in bx[j]], 'area': a})
    Image.fromarray(lab).save(OUT / f.name); json.dump(inst, open(OUT / f.name.replace('.png', '.json'), 'w')); tot += len(inst)
    s = np.array([x['score'] for x in inst]) if inst else np.zeros(0)
    print(f'[sam3] "{PROMPT}" {f.stem}: {len(inst)} masks >= {CONF} (>=0.5: {int((s >= 0.5).sum())}), best {s.max() if len(s) else 0:.2f}, median area {int(np.median([x["area"] for x in inst])) if inst else 0} px, cover {float((lab > 0).mean()):.3f}', flush=True)
print(f'[sam3] "{PROMPT}": {len(frames)} views, {tot} masks, {(time.time() - t0):.1f} s', flush=True)
