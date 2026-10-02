"""Pick a chunk side-car's verdict frame: the most-supervised keyframe whose CAMERA IS INSIDE THE CHUNK CELL (Paul, 2026-09-27;
rationale in sidecar_chunk_frame.py). v2 (2026-09-30): the side-car dir and the H3DGS project are arguments, so tagged
side-cars (chunk_0_0_expo) and non-default projects (h3dgs_expo) work; --fruit counts fruit pixels (ids >= 10000) in
supervision/trees_fruit_v3 instead of tree pixels in supervision/trees_only.
  python sidecar_chunk_frame_v2.py <survey> <chunk> [--out <side-car dir>] [--proj <h3dgs project dir>] [--fruit]  -> frame name on stdout"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('survey'); ap.add_argument('chunk'); ap.add_argument('--out', default=''); ap.add_argument('--proj', default=''); ap.add_argument('--fruit', action='store_true')
ap.add_argument('--fruit-id', type=int, default=0, help='pick the in-cell frame with the most pixels of THIS fruit id (implies --fruit)')
ap.add_argument('--tree-id', type=int, default=-1, help='pick the in-cell frame with the most pixels of THIS tree id (trees_only)'); ap.add_argument('--survey-root', default=''); a = ap.parse_args()
if a.fruit_id: a.fruit = True
S = Path(a.survey_root) if a.survey_root else Path('/home/paperspace/data/citrus_all') / a.survey; P = Path(a.proj) if a.proj else S / 'experimental/h3dgs'
O = Path(a.out) if a.out else S / 'experimental/h3dgs_sidecar_chunks' / f'chunk_{a.chunk}'
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], float)[:3, :3]
c = np.array([float(x) for x in open(P / f'camera_calibration/chunks/{a.chunk}/center.txt').read().split()])
e = np.array([float(x) for x in open(P / f'camera_calibration/chunks/{a.chunk}/extent.txt').read().split()])
lo, hi = c - e / 2, c + e / 2
tj = json.load(open(O / 'transforms.json'))
inside = set()
for f in tj['frames']:
    p = R_W @ np.asarray(f['transform_matrix'], float)[:3, 3]
    if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]: inside.add(Path(f['file_path']).name)
sup = O / ('supervision/trees_fruit_v3' if a.fruit else 'supervision/trees_only')
best = (0, None)
for f in sorted(sup.glob('*.png')):   # kf_*.png on the citrus surveys, image_N.png on phone segments
    if f.name not in inside: continue
    m = np.array(Image.open(f), np.uint16)
    n = int((m == a.tree_id).sum()) if a.tree_id >= 0 else int((m == a.fruit_id).sum()) if a.fruit_id else int(((m >= 10000) & (m != 65535)).sum()) if a.fruit else int((m != 65535).sum())
    if n > best[0]: best = (n, f.name)
print(best[1] or '')
print(f'[chunk-frame] {a.chunk}{" fruit" if a.fruit else ""}: {len(inside)}/{len(tj["frames"])} cameras inside the cell; top-supervised inside frame {best[1]} ({best[0]} px) from {sup}', file=sys.stderr)
