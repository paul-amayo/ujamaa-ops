#!/usr/bin/env python3
"""Chilli render, middle crop (2026-10-04; Paul: "do a crop for the middle section of the video, I think the PSNR on the edge
makes it worse. Then leave out the first few bad frames"). CPU only: re-cuts the saved photo | render frames of
chilli_render.py (lossless PNGs; the corner captions fall outside the crop) to the MIDDLE HALF of each panel in both
dimensions (x 270-810, y 480-1440 of 1080x1920) and re-measures PSNR on that crop.
Frames kept: image_11 .. image_141.
  - opening image_0 .. image_10 (the first few bad frames): crop PSNR 15.1 (held-out image_0) and 20.5-22.8, against
    23.1-25.8 for the trained views from image_11 on;
  - tail image_142 .. image_240: the poses collapse there (141 -> 142 jumps 7.3x the median camera step with a 17 deg turn,
    then steps shrink to ~0.04x), so even trained views render smeared at ~17 dB. A pose failure, not a render one.
Held-out views (every 10th) stay in, captioned. Outputs <out>/frames, <out>/metrics.csv, <out>/chilli_render_crop.mp4 + phone copy."""
import csv, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
SRC = Path('/home/paperspace/data/demo_video_v2/chilli/chilli_render'); OUT = Path('/home/paperspace/data/demo_video_v2/chilli/chilli_render_crop')
(OUT / 'frames').mkdir(parents=True, exist_ok=True)
W, H, CROP, FIRST, LAST = 1080, 1920, 0.5, 11, 141
x0, y0 = int(W * (1 - CROP) / 2), int(H * (1 - CROP) / 2); x1, y1 = W - x0, H - y0
rows = list(csv.reader(open(SRC / 'metrics.csv')))[1:]
F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
out, n = [], 0
for i, (name, split, full) in enumerate(rows):
    k = int(name.split('_')[1].split('.')[0])
    if not FIRST <= k <= LAST: continue
    a = np.asarray(Image.open(SRC / 'frames' / f'{i:05d}.png').convert('RGB')); ph, rd = a[y0:y1, x0:x1], a[y0:y1, W + x0:W + x1]
    psnr = float(10 * np.log10(255 ** 2 / max(((ph.astype(np.float32) - rd.astype(np.float32)) ** 2).mean(), 1e-9)))
    pim = Image.fromarray(np.concatenate([ph, rd], 1)); d = ImageDraw.Draw(pim); cw = x1 - x0
    for xo, t in ((0, 'photo (training image)'), (cw, 'H3DGS render')):
        d.rounded_rectangle([xo + 10, 10, xo + 22 + d.textlength(t, font=F1), 44], 6, fill=(11, 11, 11)); d.text((xo + 16, 14), t, font=F1, fill=(255, 255, 255))
    foot = f'{name} · ' + ('held-out' if split == 'held-out' else 'trained') + f' · {psnr:.1f} dB (middle crop; full frame {float(full):.1f})'
    d.rounded_rectangle([10, (y1 - y0) - 40, 22 + d.textlength(foot, font=F2), (y1 - y0) - 10], 6, fill=(11, 11, 11)); d.text((16, (y1 - y0) - 36), foot, font=F2, fill=(255, 255, 255))   # spans both panels (too long for one 540-px panel)
    pim.save(OUT / 'frames' / f'{n:05d}.png'); n += 1; out.append((name, split, round(psnr, 2), float(full)))
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image', 'split', 'psnr_crop_db', 'psnr_full_db']); w.writerows(out)
for s in ('trained', 'held-out'):
    v = np.array([r[2] for r in out if r[1] == s]); f = np.array([r[3] for r in out if r[1] == s])
    if v.size: print(f'[crop] {s}: {v.size} views, middle-crop PSNR min {v.min():.2f} median {np.median(v):.2f} max {v.max():.2f}, >= 25 dB {int((v >= 25).sum())} (full frame median {np.median(f):.2f})', flush=True)
for name, crf, sc in (('chilli_render_crop.mp4', 20, '1080:-2'), ('chilli_render_crop_phone.mp4', 26, '1080:-2')):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '8', '-i', str(OUT / 'frames' / '%05d.png'), '-vf', f'scale={sc},fps=24,format=yuv420p', '-c:v', 'libx264', '-crf', str(crf), '-preset', 'slow', '-movflags', '+faststart', str(OUT / name)], check=True)
    print(f'[crop] wrote {OUT / name} ({(OUT / name).stat().st_size / 2**20:.1f} MiB, {n / 8:.1f} s, {n} frames)', flush=True)
