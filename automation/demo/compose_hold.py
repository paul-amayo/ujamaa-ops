#!/usr/bin/env python3
"""Re-time a cut from its frames (2026-10-05; the dashboard session for Paul's final cut: "the Citrus B cut is too fast to read ...
re-composite citrus_b_cut at a 3 fps step with a 2 s hold on each question's answer frame"). Every frame shows for 1/fps s; the
first frame of each question segment (a run of the same mode in the metrics, any mode except 'plain') holds for --hold s.
CPU only: ffmpeg concat demuxer over the existing PNGs.
  python compose_hold.py --frames <dir> --metrics <metrics csv, 'shown' rows in frame order> --out <mp4> [--fps 3] [--hold 2]"""
import argparse, csv, subprocess, tempfile
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--frames', required=True); ap.add_argument('--metrics', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--fps', type=float, default=3.0); ap.add_argument('--hold', type=float, default=2.0); a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.metrics)) if r['status'] == 'shown']; frames = sorted(Path(a.frames).glob('*.png'))
assert len(rows) == len(frames), f'{len(rows)} shown rows vs {len(frames)} frames'
lines, prev, holds, total = ['ffconcat version 1.0'], None, 0, 0.0
for r, f in zip(rows, frames):
    d = 1.0 / a.fps
    if r['mode'] != prev and r['mode'] != 'plain': d = a.hold; holds += 1
    prev = r['mode']; total += d; lines += [f"file '{f.resolve()}'", f'duration {d:.6f}']
lines.append(f"file '{frames[-1].resolve()}'")   # the concat demuxer drops the last entry's duration
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as fh: fh.write('\n'.join(lines) + '\n'); lst = fh.name
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-crf', '20', '-movflags', '+faststart', a.out], check=True)
print(f'[hold] {len(frames)} frames at {a.fps:g} fps, {holds} answer holds of {a.hold:g} s -> {a.out} ({total:.1f} s, {Path(a.out).stat().st_size / 2**20:.1f} MiB)')
