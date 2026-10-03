#!/usr/bin/env python3
"""Per-frame PSNR of the demo reels (UJAMAA, 2026-10-03; Paul: "there are so many grey cuts, for each frame give me the
psnr graph for the video"). Every reel frame is a render at a recorded keyframe pose (demo_path.json 'src'); its PLAIN
render (backdrop/d_NNNNN.png, no identity tint, no captions) is scored against that keyframe's photo, full frame and with
the survey's sky mask (white = sky) excluded. Writes a CSV per segment."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
D = Path('/home/paperspace/logs/demo_chunks'); OUT = Path('/home/paperspace/data/demo_video_v2/psnr'); OUT.mkdir(exist_ok=True)
C = '/home/paperspace/data/citrus_all'
SEG = [  # name, reel dir, frame indices (None = all), GT dir, sky dir
    ('Citrus B lane (05 chunk 0_0)', '05_0_0_720', None, f'{C}/05_13D_Jackal/prod/scratch_sam3', f'{C}/05_13D_Jackal/prod/tassili/sky_masks'),
    ('Citrus B fruit clip (05 chunk 1_0)', '05_1_0_720', list(range(13, 53)) + list(range(346, 361)), f'{C}/05_13D_Jackal/prod/scratch_sam3', f'{C}/05_13D_Jackal/prod/tassili/sky_masks'),
    ('Citrus A (01 chunk 3_1)', '01_3_1_720', None, f'{C}/01_13B_Jackal/prod/scratch_sam3', f'{C}/01_13B_Jackal/prod/tassili/sky_masks'),
    ('Gwakungu cabbage reel', 'gwakungu_7993', None, '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root/prod/scratch_sam3', None)]
def psnr(a, b, m=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = d.mean() if m is None else d[m].mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
for name, rd, idx, gt, sky in SEG:
    P = json.load(open(D / rd / 'demo_path.json'))['frames']; frames = P if idx is None else [P[i] for i in idx]
    rows = []
    for f in frames:
        bp = D / rd / 'backdrop' / f['name']; gp = Path(gt) / f['src']
        if not bp.exists() or not gp.exists(): rows.append((f['i'], f['name'], f['src'], f.get('mode', ''), '', '', 'missing')); continue
        r = np.asarray(Image.open(bp).convert('RGB')); g = Image.open(gp).convert('RGB')
        if g.size != (r.shape[1], r.shape[0]): g = g.resize((r.shape[1], r.shape[0]), Image.BILINEAR)
        g = np.asarray(g); m = None
        if sky and (Path(sky) / f['src']).exists():
            s = np.asarray(Image.open(Path(sky) / f['src']).convert('L').resize((r.shape[1], r.shape[0]), Image.NEAREST)); m = s < 128
        rows.append((f['i'], f['name'], f['src'], f.get('mode', ''), round(float(psnr(r, g)), 2), round(float(psnr(r, g, m)), 2) if m is not None else '', ''))
    fn = OUT / f'{rd}.csv'
    with open(fn, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['i', 'frame', 'keyframe', 'mode', 'psnr_full', 'psnr_nosky', 'note']); w.writerows(rows)
    v = np.array([r[5] if r[5] != '' else r[4] for r in rows if r[4] != ''], float)
    print(f'[psnr] {name}: {len(rows)} frames | median {np.median(v):.2f} dB, min {v.min():.2f}, < 25 dB: {(v < 25).sum()} ({(v < 25).mean():.0%}), < 20 dB: {(v < 20).sum()} -> {fn}', flush=True)
