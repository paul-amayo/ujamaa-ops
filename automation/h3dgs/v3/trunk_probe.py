#!/usr/bin/env python3
"""trunk_probe.py (2026-10-20) — are trunks visible in a cell's verdict frames? SAM 3 image mode with the prompt "tree trunk" (conf 0.3,
min area 150 px, as trunk_masks.py) on the cell's frames.json frames; per frame the masks, their areas, and the tree each mask falls in
(majority supervision id inside the mask, from the cell's supervision maps), with that tree's share of the frame; writes a contact sheet
with the masks outlined.  sam3 pixi env:  trunk_probe.py <identity dir> <images dir> <out dir> [prompt="tree trunk"] [conf=0.3]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
N, IM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); PROMPT = sys.argv[4] if len(sys.argv) > 4 else 'tree trunk'; CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
OUT.mkdir(parents=True, exist_ok=True); frames = json.load(open(N / 'frames.json'))['frames']; ev = set(json.load(open(N / 'split_names.json'))['eval'])
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent
proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=CONF); rows = []; tiles = []
for f in frames:
    im = Image.open(IM / f).convert('RGB'); ids = np.array(Image.open(sup / f)); P = ids.size
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        st = proc.set_image(im); r = proc.set_text_prompt(state=st, prompt=PROMPT)
    sc = r['scores'].float().cpu().numpy() if len(r.get('scores', [])) else np.zeros(0)
    ms = r['masks'].cpu().numpy().reshape(len(sc), im.height, im.width) if len(sc) else np.zeros((0, im.height, im.width), bool)
    bx = r['boxes'].float().cpu().numpy() if len(sc) else np.zeros((0, 4)); draw = ImageDraw.Draw(im)
    for j in range(len(sc)):
        a = int(ms[j].sum())
        if a < 150: continue
        inside = ids[ms[j]]; inside = inside[(inside != 65535) & (inside < 10000)]
        tree = int(np.bincount(inside).argmax()) if len(inside) else -1; share = float((ids == tree).sum() / P) if tree >= 0 else 0.0
        rows.append({'frame': f, 'split': 'eval' if f in ev else 'train', 'score': round(float(sc[j]), 3), 'area_px': a, 'area_frac': round(a / P, 5), 'tree': tree, 'tree_frac': round(share, 4), 'in_tree_frac': round(float(len(inside) / a), 3)})
        x0, y0, x1, y1 = bx[j]; draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3); draw.text((x0 + 3, max(0, y0 - 14)), f'{sc[j]:.2f} t{tree}', fill=(255, 255, 0))
    tiles.append(im.resize((640, 360)))
json.dump(rows, open(OUT / 'trunk_probe.json', 'w'), indent=1)
sheet = Image.new('RGB', (640 * 3, 360 * ((len(tiles) + 2) // 3))); [sheet.paste(t, ((i % 3) * 640, (i // 3) * 360)) for i, t in enumerate(tiles)]; sheet.save(OUT / 'trunk_probe_sheet.jpg', quality=85)
per = {}
for r in rows: per.setdefault(r['frame'], []).append(r)
print(f'[probe] "{PROMPT}" on {len(frames)} frames: {len(rows)} masks >= 150 px; frames with >= 1 mask: {len(per)}')
for f in frames: print(f"[probe] {f} ({'eval' if f in ev else 'train'}): " + (', '.join(f"{r['area_px']} px ({100 * r['area_frac']:.2f} %) in tree {r['tree']} ({100 * r['tree_frac']:.1f} % of frame, {100 * r['in_tree_frac']:.0f} % inside)" for r in per.get(f, [])) or 'no mask'))
big = [r for r in rows if r['tree_frac'] >= 0.03]; print(f"[probe] masks inside trees >= 3 % of the frame: {len(big)}; their areas px median {np.median([r['area_px'] for r in big]) if big else 0:.0f}, max {max([r['area_px'] for r in big]) if big else 0}")
