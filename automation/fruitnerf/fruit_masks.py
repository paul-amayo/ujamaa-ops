#!/usr/bin/env python3
"""fruit_masks.py <tree_0X> — per working-set frame a uint8 mask for the HiGH trainer's HIGH_FRUIT_MASK_DIR and for the fruit census:
2 = fruit (union of the SAM3 'apple' masks of that frame), 1 = labelled non-fruit (everything else in the photo), 0 = unused. Names = the
training images (image_<k>.png). nerf_new pixi env (pycocotools)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
T = sys.argv[1]; S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); out = S / 'fruit_masks'; out.mkdir(exist_ok=True)
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
tj = json.loads((S / 'transforms.json').read_text()); W, H = tj['w'], tj['h']; n = 0; fr = []
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items():
        k = l2i[int(li)]; m = np.ones((H, W), np.uint8)
        for e in ents:
            mk = mask_utils.decode(e['rle']).astype(bool)
            if mk.shape == (H, W): m[mk] = 2
        Image.fromarray(m).save(out / f'image_{k}.png'); n += 1; fr.append(float((m == 2).mean()))
print(f'{T}: {n} fruit masks -> {out} (fruit {100 * np.mean(fr):.2f} % of the pixels on average)')
