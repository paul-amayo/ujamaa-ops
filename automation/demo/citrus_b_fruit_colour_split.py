#!/usr/bin/env python3
"""Split the field's fruit pixels that SAM3 does NOT call tree-5 fruit into fruit-coloured vs leaf-coloured (2026-10-04, follow-up to
citrus_b_fruit_sam3_vs_field.py). Colour proxy, not ground truth: per frame, CIELAB a*b* centroids of (i) SAM3's raw tree-5 fruit
pixels and (ii) the frame's tree pixels (supervision ids < 10000) outside any fruit mask and outside the field; each field-only
pixel of the photo goes to the nearer centroid. Same 13 training-view frames and masks as the comparison figure."""
import csv, glob, json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from skimage.color import rgb2lab
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; PH = f'{S}/experimental/h3dgs_expo/camera_calibration/rectified/images'
C = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'); FID, TREE = 10001, 5
res = json.load(open(C / 'fruit_sam3_vs_field.json')); want = {int(d['kf'][3:9]) for d in res}; raw = {}
for f in glob.glob(f'{S}/prod/bateleur/sam3_fruit/clip_*/frame_entries.json'):
    j = json.load(open(f)); loc = {str(fr['local_idx']): fr['kf_idx'] for fr in j['frames']}
    for li, ents in j['frame_entries'].items():
        if loc.get(li) in want: raw.setdefault(loc[li], []).extend(ents)
tot_fo = tot_fruitlike = 0; out = []
for d in res:
    fr = np.array(Image.open(C / 'frames' / f"{d['frame']:05d}.png").convert('RGB')); field = (fr[..., 0] >= 229) & (fr[..., 1] <= 26) & (fr[..., 2] >= 198)
    H, W = field.shape; r5 = np.zeros((H, W), bool); anyf = np.zeros((H, W), bool)
    for e in raw.get(int(d['kf'][3:9]), []):
        m = mu.decode(e['rle']).astype(bool); anyf |= m
        if e['parent_census_id'] == TREE: r5 |= m
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{d["kf"]}'), np.uint16); leaf = (sup < 10000) & ~anyf & ~field
    lab = rgb2lab(np.array(Image.open(f'{PH}/{d["kf"]}').convert('RGB')))[..., 1:]
    if r5.sum() < 20 or leaf.sum() < 20: continue
    cf, cl = lab[r5].mean(0), lab[leaf].mean(0); fo = field & ~r5
    x = lab[fo]; fruitlike = int((np.linalg.norm(x - cf, axis=1) < np.linalg.norm(x - cl, axis=1)).sum())
    tot_fo += int(fo.sum()); tot_fruitlike += fruitlike; out.append((d['kf'], int(fo.sum()), fruitlike, round(fruitlike / max(fo.sum(), 1), 2)))
for o in out: print(o)
print(f'[colour] field px outside SAM3 tree-5 fruit: {tot_fo:,}; fruit-coloured (nearer the SAM3 fruit centroid than the leaf centroid) {tot_fruitlike:,} = {tot_fruitlike / max(tot_fo, 1):.0%}')
