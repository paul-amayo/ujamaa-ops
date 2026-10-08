#!/usr/bin/env python3
"""Recipe v3 chunk workspace (2026-10-08): one H3DGS export chunk -> a nerfstudio splatfacto workspace with EXACTLY the views the
H3DGS chunk trained on as train_filenames (chunk images.bin minus the chunk's test.txt) and the views the H3DGS chunk evaluator scores
as test_filenames (the survey's aligned/sparse/0/test.txt entries present in the chunk, scored at the chunk's BA pose — the
--only_chunk rule of h3dgs_eval_chunk.py). Poses = the chunk's BA poses, OpenCV -> OpenGL by diag(1,-1,-1,1) (colmap_to_nerfstudio.py);
init = the chunk's points3D.ply (H3DGS's own init). Generalises the 05 1_0 one-off (splatfacto_1_0/) the A/B was measured on.
System python3 (numpy):  build_chunk_ws.py <h3dgs project> <chunk> <workspace>"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat   # noqa: E402

P, CN, WS = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
C = P / 'camera_calibration/chunks' / CN; SRC = C / 'sparse/0'
WS.mkdir(parents=True, exist_ok=True)
for link, target in (('images', P / 'camera_calibration/rectified/images'), ('sparse/0', SRC)):
    l = WS / link; l.parent.mkdir(parents=True, exist_ok=True)
    if l.is_symlink() or l.exists():
        if os.path.realpath(l) != os.path.realpath(target): raise SystemExit(f'{l} exists and points elsewhere ({os.path.realpath(l)})')
    else: l.symlink_to(target)
cam = next(iter(read_cameras_binary(str(SRC / 'cameras.bin')).values())); params = [float(p) for p in cam.params]
if cam.model in ('PINHOLE', 'OPENCV'): fx, fy, cx, cy = params[:4]; dist = params[4:8] if cam.model == 'OPENCV' else []
elif cam.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'): fx = fy = params[0]; cx, cy = params[1:3]; dist = params[3:4]
else: raise SystemExit(f'camera model {cam.model} not handled')
k1, k2, p1, p2 = (dist + [0.0] * 4)[:4]
if not (SRC / 'points3D.ply').exists():   # v3 exports (no H3DGS training) carry only points3D.bin: write the ply splatfacto seeds from
    from read_write_model import read_points3D_binary
    pts = read_points3D_binary(str(SRC / 'points3D.bin')); xyz = np.array([p.xyz for p in pts.values()], np.float32); rgb = np.array([p.rgb for p in pts.values()], np.uint8)
    with open(SRC / 'points3D.ply', 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        f.write(np.concatenate([xyz.view(np.uint8).reshape(len(xyz), 12), rgb], 1).tobytes())
by_name = {im.name: im for im in read_images_binary(str(SRC / 'images.bin')).values()}
chunk_test = {l.strip() for l in open(SRC / 'test.txt') if l.strip()}
aligned_test = [l.strip() for l in open(P / 'camera_calibration/aligned/sparse/0/test.txt') if l.strip()]
train = sorted(n for n in by_name if n not in chunk_test); test = [n for n in aligned_test if n in by_name]
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
def frame(im):
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    return {'file_path': f'images/{im.name}', 'transform_matrix': (np.linalg.inv(w2c) @ FLIP).tolist(), 'colmap_im_id': int(im.id)}
T = {'w': int(cam.width), 'h': int(cam.height), 'camera_model': 'OPENCV', 'fl_x': fx, 'fl_y': fy, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2,
     'ply_file_path': 'sparse/0/points3D.ply', 'frames': [frame(by_name[n]) for n in train + test],
     'train_filenames': [f'images/{n}' for n in train], 'test_filenames': [f'images/{n}' for n in test], 'val_filenames': [f'images/{n}' for n in test]}
json.dump(T, open(WS / 'transforms.json', 'w'), indent=1)
json.dump({'project': str(P), 'chunk': CN, 'train': train, 'test': test,
           'rule': 'train = chunk images.bin - chunk test.txt (what H3DGS trained on); test = aligned/sparse/0/test.txt present in the chunk (what h3dgs_eval_chunk.py --only_chunk scores)'},
          open(WS / 'split.json', 'w'), indent=0)
print(f'[v3 ws] {WS}: {len(by_name)} chunk images -> {len(train)} train / {len(test)} held-out; {cam.model} {cam.width}x{cam.height} '
      f'fl {fx:.1f}/{fy:.1f}; init {os.path.getsize(SRC / "points3D.ply") / 2**20:.1f} MB')
