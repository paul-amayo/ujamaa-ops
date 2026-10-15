#!/usr/bin/env python3
"""Lindendhof SAM3 prompt probe (2026-10-12; Paul: "we get a plant canopy so it'd be plant, but also look for individual tree trunks
and fruit"). The check before any lifting / census: does SAM3 image mode find the canopy ("plant"), one instance per tree
("tree trunk") and fruit ("fruit") on the lane frames, and at which confidence? Two frames per lane (30 % / 70 % along it), every
prompt at confidence 0.1; per prompt: masks at score >= 0.1 / 0.3 / 0.5, median mask area, best score. Contact sheet: one tile per
frame, masks with score >= 0.3 (plant green, trunk blue, fruit orange) + box + score.
sam3 pixi env:  sam3_prompt_probe.py <survey root> <out dir> [lanes=1,2,3,4,5]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
S, OUT = Path(sys.argv[1]), Path(sys.argv[2]); LANES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4,5').split(',')]; OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = [('plant', (40, 200, 60)), ('tree trunk', (40, 110, 230)), ('fruit', (255, 140, 0))]
if os.environ.get('PROMPTS'):   # PROMPTS="tree,tree canopy,foliage" (2026-10-12 second probe: "plant" fired on inter-row weeds, not the canopy)
    _cols = [(40, 200, 60), (230, 60, 200), (240, 220, 40), (40, 110, 230), (255, 140, 0)]
    PROMPTS = [(p.strip(), _cols[i % len(_cols)]) for i, p in enumerate(os.environ['PROMPTS'].split(','))]
proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.1)
frames = []
for n in LANES:
    imgs = sorted((S / f'experimental/lane{n}/images').glob('f_*.png'))
    if imgs: frames += [(n, imgs[int(len(imgs) * q)]) for q in (0.3, 0.7)]
rows, tiles = [], []
for lane, p in frames:
    im = Image.open(p).convert('RGB'); ov = im.copy(); dr = ImageDraw.Draw(ov, 'RGBA'); row = {'lane': lane, 'frame': p.name}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        st = proc.set_image(im)
        for prompt, col in PROMPTS:
            proc.reset_all_prompts(st); r = proc.set_text_prompt(state=st, prompt=prompt)
            sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0)
            ms = r['masks'].cpu().numpy().reshape(len(sc), im.height, im.width) if len(sc) else np.zeros((0, im.height, im.width), bool)
            bx = r['boxes'].float().cpu().numpy() if len(sc) else np.zeros((0, 4))
            area = ms.reshape(len(sc), -1).sum(1) if len(sc) else np.zeros(0)
            row[prompt] = {f'n>={t}': int((sc >= t).sum()) for t in (0.1, 0.3, 0.5)}
            row[prompt].update(best=round(float(sc.max()), 3) if len(sc) else None, median_area_px=int(np.median(area[sc >= 0.3])) if (sc >= 0.3).any() else None,
                               cover_frac=round(float(ms[sc >= 0.3].any(0).mean()), 3) if (sc >= 0.3).any() else 0.0)
            for m, b, s in zip(ms, bx, sc):
                if s < 0.3: continue
                a = Image.fromarray((m * 90).astype(np.uint8)); ov.paste(Image.new('RGB', im.size, col), (0, 0), a)
                dr.rectangle([float(v) for v in b], outline=col + (255,), width=3); dr.text((float(b[0]) + 4, float(b[1]) + 4), f'{prompt[:5]} {s:.2f}', fill=col + (255,))
    dr.text((10, 10), f'lane {lane} {p.name}', fill=(255, 255, 255, 255)); tiles.append(ov.resize((640, 360))); rows.append(row)
    print(f"[probe] lane {lane} {p.name}: " + '; '.join(f"{k} {json.dumps(v)}" for k, v in row.items() if isinstance(v, dict)), flush=True)
cols = 2; sheet = Image.new('RGB', (640 * cols, 360 * ((len(tiles) + cols - 1) // cols)), (0, 0, 0))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 640, (i // cols) * 360))
sheet.save(OUT / 'contact_sheet.jpg', quality=88); json.dump(rows, open(OUT / 'probe.json', 'w'), indent=1)
for prompt, _ in PROMPTS:
    v = [r[prompt] for r in rows]
    print(f"[probe] {prompt!r}: frames with >= 1 mask at 0.3: {sum(1 for x in v if x['n>=0.3'])}/{len(v)}; masks/frame at 0.3 median {np.median([x['n>=0.3'] for x in v]):.0f}, "
          f"at 0.5 median {np.median([x['n>=0.5'] for x in v]):.0f}; best score median {np.median([x['best'] or 0 for x in v]):.2f}; cover median {np.median([x['cover_frac'] for x in v]):.2f}", flush=True)
