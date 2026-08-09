#!/usr/bin/env python3
"""DEC-2025 klapmuts detection-ledger sweep (Paul 2026-08-08: night queue
stopped in lieu of this). For every extracted Dec frame, run SAM3 image mode
with the three recipe prompts — 'plant pot' (identity anchor; 'sack'/'bag'
detect nothing on this epoch), 'plant' (berry parent; containment measured
92.9% berry px on dec_0144), 'berry' — and save per-frame masks+scores to
sam3_ledger_v0/<frame>.npz. This is the ingest-independent half of the Dec
fruit level: when the mcap ingest lands, these ledgers associate to the
census in 3D without re-running SAM3.

Resumable: frames with an existing npz are skipped.
"""
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, '/home/paperspace/code/sam3')
from sam3.model_builder import build_sam3_image_model       # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor   # noqa: E402

FRAMES = Path('/home/paperspace/data/klapmuts/dec_2025_a300/frames')
OUT = Path('/home/paperspace/data/klapmuts/dec_2025_a300/sam3_ledger_v0')
OUT.mkdir(exist_ok=True)
PROMPTS = {'pot': 'plant pot', 'plant': 'plant', 'berry': 'berry'}

model = build_sam3_image_model()
proc = Sam3Processor(model)
print('[sweep] SAM3 ready', flush=True)

frames = sorted(FRAMES.glob('dec_*.png'))
done = 0
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    for f in frames:
        dst = OUT / (f.stem + '.npz')
        if dst.exists():
            continue
        img = Image.open(f).convert('RGB')
        payload = {}
        for key, prompt in PROMPTS.items():
            state = proc.set_image(img)
            out = proc.set_text_prompt(state=state, prompt=prompt)
            ms, sc = out.get('masks'), out.get('scores')
            scl = (sc.float().cpu().numpy() if hasattr(sc, 'cpu')
                   else np.array(sc or [], dtype=np.float32))
            keep_m, keep_s = [], []
            if ms is not None:
                for i, m in enumerate(ms):
                    if i < len(scl) and scl[i] >= 0.3:
                        a = (m.detach().cpu().numpy() if hasattr(m, 'detach')
                             else np.asarray(m)).squeeze().astype(bool)
                        keep_m.append(a)
                        keep_s.append(float(scl[i]))
            payload[f'{key}_masks'] = (np.stack(keep_m) if keep_m
                                       else np.zeros((0, img.size[1], img.size[0]), bool))
            payload[f'{key}_scores'] = np.array(keep_s, np.float32)
        np.savez_compressed(dst, **payload)
        done += 1
        if done % 20 == 0:
            print(f'[sweep] {done} frames done', flush=True)
print(f'SWEEP-DONE: {done} new frames -> {OUT}', flush=True)
