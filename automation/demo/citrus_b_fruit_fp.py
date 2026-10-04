#!/usr/bin/env python3
"""Fruit containment vs SAM3 per BLOB, with false positives (2026-10-04; Paul: "this is not clear, also are there any false positives").
CPU only; nerf_new pixi env (pycocotools). Citrus B training views: the 13 fruit frames of citrus_b_cut_trained (expo chunk 1_0).
  field  tree 5's fruit (10001) by best containment = the cut's magenta pixels (exact: 0.1 render + 0.9 magenta), on the VISIBLE
         part of the frame: the caption boxes (exact fill colours, components > 500 px, their bounding boxes) are removed from BOTH
         the field and SAM3 before anything is counted;
  SAM3   the raw "fruit" detections (run inside each census tree's crop; parent_census_id 5 = tree 5).
Each field blob (8-connected) gets ONE verdict:
  TRUE POSITIVE   touches a SAM3 tree-5 fruit mask;
  WRONG TREE      touches only another tree's SAM3 fruit;
  NO SAM3 FRUIT   touches no SAM3 fruit -> split by colour (proxy, not ground truth): CIELAB a*b* of the blob's photo pixels nearer
                  the frame's SAM3 fruit pixels ("looks like an orange SAM3 missed") or nearer the frame's tree pixels
                  ("leaf: false positive").
Each SAM3 tree-5 fruit mask: FOUND if any field pixel touches it, else MISSED. Bleed = field pixels of true-positive blobs outside
the SAM3 masks. Figure: 4 frames, photo crop | same crop with verdicts. -> citrus_b_cut_trained/fruit_false_positives.{jpg,json}"""
import csv, glob, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
from scipy import ndimage as nd
from skimage.color import rgb2lab
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; PH = f'{S}/experimental/h3dgs_expo/camera_calibration/rectified/images'
C = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'); TREE = 5
score = {r['kf']: r for r in json.load(open(f'{NB}/fruit_bc_baseline.json'))}
rows = list(csv.DictReader(open(C / 'metrics.csv')))
sel = [(n, r['keyframe']) for n, r in enumerate(rows) if r['mode'] == 'fruit' and score.get(r['keyframe'], {}).get('iou') is not None]
want = {int(kf[3:9]) for _, kf in sel}; raw = {}
for f in glob.glob(f'{S}/prod/bateleur/sam3_fruit/clip_*/frame_entries.json'):
    j = json.load(open(f)); loc = {str(fr['local_idx']): fr['kf_idx'] for fr in j['frames']}
    for li, ents in j['frame_entries'].items():
        if loc.get(li) in want: raw.setdefault(loc[li], []).extend(ents)
EIGHT = np.ones((3, 3), bool); T = dict(tp=0, wrong=0, orange=0, leaf=0, found=0, missed=0, lit=0, bleed=0, leaf_px=0, orange_px=0); per, keep = [], {}
for n, kf in sel:
    fr = np.array(Image.open(C / 'frames' / f'{n:05d}.png').convert('RGB')); H, W = fr.shape[:2]; hidden = np.zeros((H, W), bool)
    for col in ((11, 11, 11), (255, 122, 0)):
        lab, k = nd.label((fr == col).all(-1))
        for i, sl in enumerate(nd.find_objects(lab)):
            if (lab[sl] == i + 1).sum() > 500: hidden[sl] = True
    vis = ~hidden; field = (fr[..., 0] >= 229) & (fr[..., 1] <= 26) & (fr[..., 2] >= 198) & vis
    m5 = []; other = np.zeros((H, W), bool)
    for e in raw.get(int(kf[3:9]), []):
        m = mu.decode(e['rle']).astype(bool) & vis
        if not m.any(): continue
        if e['parent_census_id'] == TREE: m5.append(m)
        else: other |= m
    s5 = np.any(m5, 0) if m5 else np.zeros((H, W), bool)
    photo = np.array(Image.open(f'{PH}/{kf}').convert('RGB')); ab = rgb2lab(photo)[..., 1:]
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16); treepx = (sup < 10000) & ~s5 & ~other & ~field & vis
    cf = ab[s5].mean(0) if s5.sum() >= 20 else None; cl = ab[treepx].mean(0)
    lab, nb = nd.label(field, EIGHT); verdict = np.zeros(nb + 1, np.int8)   # 1 tp, 2 wrong tree, 3 orange-coloured, 4 leaf
    for b in range(1, nb + 1):
        bm = lab == b
        if (bm & s5).any(): verdict[b] = 1
        elif (bm & other).any(): verdict[b] = 2
        else:
            c = ab[bm].mean(0); verdict[b] = 3 if cf is not None and np.linalg.norm(c - cf) < np.linalg.norm(c - cl) else 4
    v = verdict[lab]; found = sum(bool((m & field).any()) for m in m5)
    d = dict(kf=kf, blobs=int(nb), tp=int((verdict == 1).sum()), wrong_tree=int((verdict == 2).sum()), orange_no_sam3=int((verdict == 3).sum()), leaf_fp=int((verdict == 4).sum()),
             sam3_fruit=len(m5), found=found, missed=len(m5) - found, lit_px=int(field.sum()), bleed_px=int((field & (v == 1) & ~s5).sum()),
             orange_px=int((v == 3).sum()), leaf_px=int((v == 4).sum()), hidden_px=int(hidden.sum()))
    per.append(d); keep[kf] = (n, field, s5, m5, v, photo)
    for a, b in (('tp', 'tp'), ('wrong', 'wrong_tree'), ('orange', 'orange_no_sam3'), ('leaf', 'leaf_fp'), ('found', 'found'), ('missed', 'missed'), ('lit', 'lit_px'), ('bleed', 'bleed_px'), ('leaf_px', 'leaf_px'), ('orange_px', 'orange_px')): T[a] += d[b]
    print(d, flush=True)
nb_all = T['tp'] + T['wrong'] + T['orange'] + T['leaf']
print(f"[fp] 13 frames, {nb_all} field blobs: {T['tp']} touch SAM3 tree-5 fruit, {T['wrong']} touch only another tree's fruit, {T['orange']} orange-coloured with no SAM3 fruit, "
      f"{T['leaf']} leaf-coloured with no SAM3 fruit (FALSE POSITIVES). SAM3 tree-5 fruit: {T['found']} found, {T['missed']} missed. "
      f"Pixels: {T['lit']:,} lit = TP blobs {T['lit'] - T['orange_px'] - T['leaf_px']:,} (of which bleed outside SAM3 {T['bleed']:,}) + orange-coloured {T['orange_px']:,} + leaf FP {T['leaf_px']:,}")
json.dump({'per_frame': per, 'totals': T}, open(C / 'fruit_false_positives.json', 'w'), indent=1)
# ---- figure: 4 frames, photo crop | verdict overlay; crop = the 480 x 270 window holding the most field + SAM3 pixels, shown 2x
F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
COL = {1: (40, 210, 40), 2: (60, 120, 255), 3: (255, 200, 0), 4: (235, 30, 30)}; cw, chh, Z = 480, 270, 2
strips = []
for kf in ('kf_000415.png', 'kf_000025.png', 'kf_003282.png', 'kf_003284.png'):
    n, field, s5, m5, v, photo = keep[kf]; d = next(x for x in per if x['kf'] == kf); H, W = field.shape
    dens = nd.uniform_filter((field | s5).astype(np.float32), size=(chh, cw), mode='constant'); cy, cx = np.unravel_index(dens.argmax(), dens.shape)
    y0, x0 = int(np.clip(cy - chh // 2, 0, H - chh)), int(np.clip(cx - cw // 2, 0, W - cw)); box = (x0, y0, x0 + cw, y0 + chh)
    ov = photo.astype(np.float32).copy()
    for k, col in COL.items(): m = v == k; ov[m] = 0.35 * ov[m] + 0.65 * np.array(col)
    ov = ov.astype(np.uint8); outline = nd.binary_dilation(s5, iterations=1) & ~nd.binary_erosion(s5, iterations=1); ov[outline] = (0, 235, 255)
    a = Image.fromarray(photo).crop(box).resize((cw * Z, chh * Z), Image.LANCZOS); b = Image.fromarray(ov).crop(box).resize((cw * Z, chh * Z), Image.NEAREST)
    strip = Image.new('RGB', (cw * Z * 2 + 20, chh * Z + 44), (255, 255, 255)); strip.paste(a, (0, 44)); strip.paste(b, (cw * Z + 20, 44))
    ImageDraw.Draw(strip).text((6, 8), f"{kf}:  {d['tp']} correct blobs (green), {d['leaf_fp']} false positives on leaves (red), {d['orange_no_sam3']} on oranges SAM3 did not mask (yellow); SAM3 fruit found {d['found']} / missed {d['missed']}", font=F2, fill=(0, 0, 0))
    strips.append(strip)
head1 = 'Tree 5 fruit on Citrus B training views: left = photo, right = what containment lit'
head2 = 'green = lit on a SAM3 fruit · red = lit on leaves, no fruit (FALSE POSITIVE) · yellow = lit on an orange SAM3 missed (colour test) · cyan outline = SAM3 fruit mask'
tot = (f"all 13 frames: {nb_all} lit blobs = {T['tp']} correct, {T['leaf']} false positives on leaves, {T['orange']} on oranges SAM3 missed, {T['wrong']} on another tree's fruit; "
       f"SAM3 fruit found {T['found']} of {T['found'] + T['missed']}")
Wd = strips[0].width; sheet = Image.new('RGB', (Wd, sum(s.height for s in strips) + 120), (255, 255, 255)); dd = ImageDraw.Draw(sheet)
dd.text((6, 6), head1, font=F1, fill=(0, 0, 0)); dd.text((6, 44), head2, font=F2, fill=(0, 0, 0)); dd.text((6, 78), tot, font=F2, fill=(0, 0, 0)); y = 120
for s in strips: sheet.paste(s, (0, y)); y += s.height
sheet.save(C / 'fruit_false_positives.jpg', quality=90); print(f"[fp] -> {C / 'fruit_false_positives.jpg'} {sheet.size}")
