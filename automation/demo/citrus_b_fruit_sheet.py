#!/usr/bin/env python3
"""What fruit containment looks like on Citrus B's TRAINING views (2026-10-04; Paul: "what does fruit containment look like"). CPU only.
Every fruit-segment frame of citrus_b_cut_trained (expo chunk 1_0, cell chunk_1_0_sam3: tree 5's oranges, fruit 10001, lit in
magenta + white edge where the fruit word beats every other bank word) that SAM3 labels with fruit 10001, with SAM3's fruit mask
outlined in cyan on top. Ordered by IoU (fruit_bc_score baseline json: same field, same rule). -> fruit_containment_trained.jpg"""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
NB = '/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3'; FID = 10001
C = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'); OUT = C / 'fruit_containment_trained.jpg'
score = {r['kf']: r for r in json.load(open(f'{NB}/fruit_bc_baseline.json'))}
rows = list(csv.DictReader(open(C / 'metrics.csv')))
sel = [(n, r) for n, r in enumerate(rows) if r['mode'] == 'fruit' and score.get(r['keyframe'], {}).get('iou') is not None]
sel.sort(key=lambda t: -score[t[1]['keyframe']]['iou'])
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
tiles = []
for n, r in sel:
    im = Image.open(C / 'frames' / f'{n:05d}.png').convert('RGB'); W, H = im.size
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{r["keyframe"]}'), np.uint16)
    if sup.shape != (H, W): sup = np.array(Image.fromarray(sup).resize((W, H), Image.NEAREST))
    g = sup == FID; edge = nd.binary_dilation(g, iterations=2) & ~g
    a = np.array(im); a[edge] = (0, 230, 255); im = Image.fromarray(a); d = ImageDraw.Draw(im); s = score[r['keyframe']]
    t = f"{r['keyframe']}  IoU {s['iou']:.2f}  field {s['lit']:,} px / SAM3 {s['gt']:,} px"
    d.rounded_rectangle([12, H - 52, 26 + d.textlength(t, font=F), H - 14], 8, fill=(11, 11, 11)); d.text((19, H - 47), t, font=F, fill=(255, 255, 255))
    tiles.append(im.resize((W // 2, H // 2)))
cols = 3; rws = (len(tiles) + cols - 1) // cols; w, h = tiles[0].size
sheet = Image.new('RGB', (cols * w, rws * h + 50), (255, 255, 255)); dd = ImageDraw.Draw(sheet)
dd.text((12, 12), f'Citrus B training views, fruit 10001 (tree 5): field = magenta + white edge, SAM3 = cyan outline; {len(tiles)} frames, mean IoU {np.mean([score[r["keyframe"]]["iou"] for _, r in sel]):.3f}', font=F, fill=(0, 0, 0))
for k, t in enumerate(tiles): sheet.paste(t, ((k % cols) * w, 50 + (k // cols) * h))
sheet.save(OUT, quality=88); print(f'[fruit] {len(tiles)} frames -> {OUT}; IoUs', [round(score[r["keyframe"]]["iou"], 2) for _, r in sel])
