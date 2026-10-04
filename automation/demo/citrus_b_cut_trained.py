#!/usr/bin/env python3
"""Citrus B cut, TRAINING VIEWS ONLY (2026-10-04; Paul: "so if we use citrus B training views what does that look like"). CPU only:
keeps the frames of citrus_b_cut_full (expo chunk 1_0, best containment, every path frame) whose keyframe is a training view of
the chunk (not in test.txt, has a trained exposure) - no PSNR minimum - and re-encodes them in path order. Outputs
data/demo_video_v2/citrus_b_cut_trained/{frames, metrics.csv, citrus_b_cut_trained.mp4, citrus_b_cut_trained_phone.mp4}."""
import csv, json, os, subprocess
from pathlib import Path
P = '/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo'; SRC = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_full')
OUT = Path('/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
TEST = {l.strip() for l in open(f'{P}/camera_calibration/chunks/1_0/sparse/0/test.txt') if l.strip()}; EXPO = json.load(open(f'{P}/output/trained_chunks/1_0/exposure.json'))
rows = list(csv.DictReader(open(SRC / 'metrics.csv'))); assert all(r['status'] == 'shown' for r in rows) and len(rows) == len(list((SRC / 'frames').glob('*.png')))
keep = [(i, r) for i, r in enumerate(rows) if r['keyframe'] not in TEST and r['keyframe'] in EXPO]
for n, (i, r) in enumerate(keep):
    dst = OUT / 'frames' / f'{n:05d}.png'
    if not dst.exists(): os.link(SRC / 'frames' / f'{i:05d}.png', dst)
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(r for _, r in keep)
for name, crf, sc in (('citrus_b_cut_trained.mp4', 20, 'iw:ih'), ('citrus_b_cut_trained_phone.mp4', 28, '1280:-2')):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '8', '-i', str(OUT / 'frames' / '%05d.png'), '-vf', f'scale={sc},fps=24,format=yuv420p', '-c:v', 'libx264', '-crf', str(crf), '-preset', 'slow', '-movflags', '+faststart', str(OUT / name)], check=True)
    print(f'[trained] {name}: {(OUT / name).stat().st_size / 2**20:.1f} MiB, {len(keep)} frames, {len(keep) / 8:.1f} s', flush=True)
