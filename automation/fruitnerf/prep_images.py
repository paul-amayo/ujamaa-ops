#!/usr/bin/env python3
"""prep_images.py <tree_0X> — the working set for fruitnerf_structure.sh in parallel (JPEG decode of 6000x4000 is the cost): all frames, height
1080 (1620x1080 PNG), sequential image_<i>.png in sorted source order, source_names.txt. Writes to images.tmp/ then renames, so the structure
script (which skips a non-empty images/) never sees a partial set."""
import sys, os
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
T = sys.argv[1]; src = Path(f'/home/paperspace/data/public/fruitnerf/FruitNeRF_Dataset/{T}/images'); S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}')
frames = sorted(src.glob('*.JPG')) + sorted(src.glob('*.jpg')); tmp = S / 'images.tmp'; tmp.mkdir(parents=True, exist_ok=True)
def one(a):
    i, p = a; out = tmp / f'image_{i}.png'
    if out.exists(): return
    im = Image.open(p); im.draft('RGB', (im.size[0] // 2, im.size[1] // 2)); im = im.convert('RGB')   # JPEG draft decode at 1/2 res, then LANCZOS to 1080
    w, h = Image.open(p).size; nw = int(round(w * 1080 / h / 2) * 2); im.resize((nw, 1080), Image.LANCZOS).save(out)
with Pool(12) as pool: pool.map(one, list(enumerate(frames)))
(S / 'source_names.txt').write_text('\n'.join(p.name for p in frames) + '\n')
if not (S / 'images').exists() or not any((S / 'images').iterdir()):
    if (S / 'images').exists(): (S / 'images').rmdir()
    os.rename(tmp, S / 'images')
print(f'{T}: {len(frames)} frames ready')
