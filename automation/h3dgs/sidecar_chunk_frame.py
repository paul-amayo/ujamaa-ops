"""Pick a chunk side-car's verdict frame: the most-supervised keyframe whose CAMERA IS INSIDE THE CHUNK CELL (Paul, 2026-09-27).
A chunk's nerfstudio dataset carries every camera H3DGS's chunk BA used, and ~40% of those sit outside the cell (0_1: 488 of
1222) — they legitimately supervise the census because they see into the cell, but the side-car's leaves are cropped to
cell + margin, so scoring on one of them measures geometry that was deliberately cut away. Chunk 0_1's first verdict frame
kf_002166 had its camera 11.0 m beyond the cell edge: trees 0.367/0.391, rows 0.449/0.359 at prec ~0.55 / rec ~0.52 (half a
mask), against 0.691/0.923 and 0.516/0.793 on 0_0, whose frame was inside.
  python sidecar_chunk_frame.py <survey> <chunk>   -> frame name on stdout"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sv, cn = sys.argv[1], sys.argv[2]
S = Path('/home/paperspace/data/citrus_all') / sv; P = S / 'experimental/h3dgs'; O = S / 'experimental/h3dgs_sidecar_chunks' / f'chunk_{cn}'
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], float)[:3, :3]
c = np.array([float(x) for x in open(P / f'camera_calibration/chunks/{cn}/center.txt').read().split()])
e = np.array([float(x) for x in open(P / f'camera_calibration/chunks/{cn}/extent.txt').read().split()])
lo, hi = c - e / 2, c + e / 2
tj = json.load(open(O / 'transforms.json'))
inside = set()
for f in tj['frames']:
    p = R_W @ np.asarray(f['transform_matrix'], float)[:3, 3]
    if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]: inside.add(Path(f['file_path']).name)
best = (0, None)
for f in sorted((O / 'supervision/trees_only').glob('kf_*.png')):
    if f.name not in inside: continue
    n = int((np.array(Image.open(f), np.uint16) != 65535).sum())
    if n > best[0]: best = (n, f.name)
print(best[1] or '')
print(f'[chunk-frame] {cn}: {len(inside)}/{len(tj["frames"])} cameras inside the cell; top-supervised inside frame {best[1]} ({best[0]} px)', file=sys.stderr)
