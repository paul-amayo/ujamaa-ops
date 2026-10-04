#!/usr/bin/env python3
"""SAM3 fruit masks against fruit containment on Citrus B's training views (2026-10-04; Paul: "sam3 masks on fruits against
containment"). CPU only; run in the nerf_new pixi env (pycocotools).
For the 13 training-view fruit frames of citrus_b_cut_trained (expo chunk 1_0, cell chunk_1_0_sam3):
  field   tree 5's fruit word (10001) winning best containment over every bank word = the cut's magenta pixels, recovered
          exactly from the saved frame (lit = 0.1 render + 0.9 (255, 0, 220): R >= 229, G <= 26, B >= 198; checked against
          the scorer's lit count);
  SAM3    the RAW "fruit" detections (prod/bateleur/sam3_fruit, run inside each census tree's crop; parent_census_id = the tree),
          and the supervision's fruit 10001 (what the scorer and the 0.355 mean use: the raw masks after strict_fruit_tree_v1
          and the painter).
Per frame: IoU field vs raw SAM3 tree-5 fruit, IoU field vs supervision 10001, share of lit px on ANY raw SAM3 fruit, recall of
raw tree-5 fruit. Figure rows (6 frames across the IoU range), zoomed on the fruit: photo + raw SAM3 fruit (cyan = tree 5,
yellow = other trees) | render + field (magenta) | agreement on the photo (green both, magenta field only, cyan SAM3 tree-5 only,
yellow field on another tree's SAM3 fruit). -> citrus_b_cut_trained/fruit_sam3_vs_field.jpg + .json"""
import csv, glob, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
from scipy import ndimage as nd
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; PH = f'{S}/experimental/h3dgs_expo/camera_calibration/rectified/images'
C = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'); FID, TREE = 10001, 5
score = {r['kf']: r for r in json.load(open(f'{NB}/fruit_bc_baseline.json'))}
rows = list(csv.DictReader(open(C / 'metrics.csv')))
sel = [(n, r['keyframe']) for n, r in enumerate(rows) if r['mode'] == 'fruit' and score.get(r['keyframe'], {}).get('iou') is not None]
want = {int(kf[3:9]) for _, kf in sel}; raw = {}
for f in glob.glob(f'{S}/prod/bateleur/sam3_fruit/clip_*/frame_entries.json'):
    j = json.load(open(f)); loc = {str(fr['local_idx']): fr['kf_idx'] for fr in j['frames']}
    for li, ents in j['frame_entries'].items():
        if loc.get(li) in want: raw.setdefault(loc[li], []).extend(ents)
def iou(a, b): return float((a & b).sum() / max((a | b).sum(), 1))
res = []
for n, kf in sel:
    fr = np.array(Image.open(C / 'frames' / f'{n:05d}.png').convert('RGB')); H, W = fr.shape[:2]
    field = (fr[..., 0] >= 229) & (fr[..., 1] <= 26) & (fr[..., 2] >= 198)
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16); s5 = sup == FID
    r5, ro = np.zeros((H, W), bool), np.zeros((H, W), bool); n5 = no = 0
    for e in raw.get(int(kf[3:9]), []):
        m = mu.decode(e['rle']).astype(bool)
        if e['parent_census_id'] == TREE: r5 |= m; n5 += 1
        else: ro |= m; no += 1
    res.append(dict(kf=kf, frame=n, field_px=int(field.sum()), scorer_lit=score[kf]['lit'], raw5_n=n5, raw5_px=int(r5.sum()), rawother_n=no, sup5_px=int(s5.sum()),
                    iou_raw5=round(iou(field, r5), 3), iou_sup5=round(iou(field, s5), 3), scorer_iou=round(score[kf]['iou'], 3),
                    lit_on_raw5=round(float((field & r5).sum() / max(field.sum(), 1)), 3), lit_on_rawother=round(float((field & ro & ~r5).sum() / max(field.sum(), 1)), 3),
                    recall_raw5=round(float((field & r5).sum() / max(r5.sum(), 1)), 3), sup5_of_raw5=round(float((s5 & r5).sum() / max(r5.sum(), 1)), 3)))
    res[-1]['_m'] = (field, r5, ro, s5)
res.sort(key=lambda d: -d['scorer_iou'])
for d in res: print({k: v for k, v in d.items() if k != '_m'})
mean = lambda k: float(np.mean([d[k] for d in res]))
print(f"[fruit] {len(res)} frames: IoU field vs raw SAM3 tree-5 fruit {mean('iou_raw5'):.3f}, vs supervision {mean('iou_sup5'):.3f} (scorer {mean('scorer_iou'):.3f}); "
      f"lit px on raw tree-5 fruit {mean('lit_on_raw5'):.2f}, on other trees' fruit {mean('lit_on_rawother'):.2f}; recall of raw tree-5 fruit {mean('recall_raw5'):.2f}; "
      f"supervision keeps {mean('sup5_of_raw5'):.2f} of the raw tree-5 fruit px")
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18); TH = 300
pick = [res[i] for i in sorted({0, 2, 5, 8, 10, len(res) - 1}) if i < len(res)]; strips = []
for d in pick:
    field, r5, ro, s5 = d['_m']; kf = d['kf']; H, W = field.shape
    ys, xs = np.nonzero(field | r5); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max(x1 - x0 + 60, 320); h = max(y1 - y0 + 60, 180); w, h = max(w, h * 16 / 9), max(h, w * 9 / 16)
    bx0, by0 = int(np.clip(cx - w / 2, 0, max(W - w, 0))), int(np.clip(cy - h / 2, 0, max(H - h, 0))); box = (bx0, by0, int(min(bx0 + w, W)), int(min(by0 + h, H)))
    photo = np.array(Image.open(f'{PH}/{kf}').convert('RGB')); render = np.array(Image.open(C / 'frames' / f"{d['frame']:05d}.png").convert('RGB'))
    p1 = photo.copy()
    for m, col in ((ro & ~r5, (255, 220, 0)), (r5, (0, 230, 255))):
        p1[m] = (0.35 * p1[m] + 0.65 * np.array(col)).astype(np.uint8); e = nd.binary_dilation(m, iterations=1) & ~m; p1[e] = col
    p3 = (photo * 0.45).astype(np.uint8)
    for m, col in ((field & ~r5 & ~ro, (255, 0, 220)), (r5 & ~field, (0, 230, 255)), (field & ro & ~r5, (255, 220, 0)), (field & r5, (40, 220, 40))): p3[m] = col
    panels = [Image.fromarray(a).crop(box) for a in (p1, render, p3)]; sc = TH / panels[0].height; panels = [p.resize((int(p.width * sc), TH), Image.NEAREST) for p in panels]
    strip = Image.new('RGB', (sum(p.width for p in panels) + 20, TH + 30), (255, 255, 255)); x = 0
    for p in panels: strip.paste(p, (x, 30)); x += p.width + 10
    ImageDraw.Draw(strip).text((4, 4), f"{kf}: IoU vs raw SAM3 tree-5 fruit {d['iou_raw5']:.2f} (vs supervision {d['iou_sup5']:.2f}); raw tree-5 fruit {d['raw5_n']} masks / {d['raw5_px']:,} px, supervision keeps {d['sup5_px']:,} px; field {d['field_px']:,} px, {d['lit_on_raw5']:.0%} on raw tree-5 fruit", font=F, fill=(0, 0, 0))
    strips.append(strip)
head = 'SAM3 fruit vs containment, Citrus B training views (zoomed): photo + raw SAM3 "fruit" (cyan tree 5, yellow other trees) | render + field (magenta) | agreement: green both, magenta field only, cyan SAM3 only, yellow field on another tree\'s fruit'
Wd = max(s.width for s in strips); sheet = Image.new('RGB', (Wd, sum(s.height for s in strips) + 40), (255, 255, 255)); ImageDraw.Draw(sheet).text((4, 10), head, font=F, fill=(0, 0, 0)); y = 40
for s in strips: sheet.paste(s, (0, y)); y += s.height
sheet.save(C / 'fruit_sam3_vs_field.jpg', quality=88); json.dump([{k: v for k, v in d.items() if k != '_m'} for d in res], open(C / 'fruit_sam3_vs_field.json', 'w'), indent=1)
print(f"[fruit] -> {C / 'fruit_sam3_vs_field.jpg'} ({sheet.size[0]}x{sheet.size[1]})")
