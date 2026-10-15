#!/usr/bin/env python3
"""fruitnerf_poses.py <tree_0X> — use FruitNeRF's own COLMAP poses (our GLOMAP registered only 194/366 and 220/422 on trees 02/03, 10-14):
undistort the working set (their OPENCV k1/k2/p1/p2, intrinsics scaled to the 1080-high working size, new pinhole K from
getOptimalNewCameraMatrix alpha 0 = no black corners; raw kept in images_raw/), and write transforms.json (PINHOLE, c2w OpenGL, file_path
images/image_<k>.png for the posed frames) + lio_image_poses.json (c2w OpenCV). Our own SfM outputs of tree_01 are moved aside (*_oursfm).
miniconda python3 (cv2):  fruitnerf_poses.py tree_01"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, cv2
T = sys.argv[1]; S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); src = json.loads(Path(f'/home/paperspace/data/public/fruitnerf/FruitNeRF_Dataset/{T}/transforms.json').read_text())
names = (S / 'source_names.txt').read_text().split(); k_of = {n: k for k, n in enumerate(names)}
for f in ('transforms.json', 'lio_image_poses.json', 'lio_image_poses_kf20cm.json'):
    if (S / f).exists() and not (S / f.replace('.json', '_oursfm.json')).exists(): os.rename(S / f, S / f.replace('.json', '_oursfm.json'))
raw = S / 'images_raw'
if not raw.exists(): os.rename(S / 'images', raw); (S / 'images').mkdir()
im0 = cv2.imread(str(raw / 'image_0.png')); H, W = im0.shape[:2]; s = H / src['h']
K = np.array([[src['fl_x'] * s, 0, src['cx'] * s], [0, src['fl_y'] * s, src['cy'] * s], [0, 0, 1]]); dist = np.array([src['k1'], src['k2'], src['p1'], src['p2']])
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0, (W, H)); mx, my = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_32FC1)
for p in sorted(raw.glob('image_*.png')):
    out = S / 'images' / p.name
    if not out.exists(): cv2.imwrite(str(out), cv2.remap(cv2.imread(str(p)), mx, my, cv2.INTER_LINEAR))
frames, lio = [], []
for fr in src['frames']:
    k = k_of.get(Path(fr['file_path']).name)
    if k is None: continue
    gl = np.asarray(fr['transform_matrix'], float); cv_ = gl.copy(); cv_[:3, 1] *= -1; cv_[:3, 2] *= -1
    frames.append({'file_path': f'images/image_{k}.png', 'transform_matrix': gl.tolist()}); lio.append({'image_idx': k, 'image_name': f'image_{k}.png', 'timestamp_ns': k * 500000000, 'transform': cv_.tolist()})
frames.sort(key=lambda f: int(f['file_path'].split('_')[-1].split('.')[0])); lio.sort(key=lambda e: e['image_idx'])
tj = {'w': W, 'h': H, 'fl_x': float(newK[0, 0]), 'fl_y': float(newK[1, 1]), 'cx': float(newK[0, 2]), 'cy': float(newK[1, 2]), 'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
      'camera_model': 'PINHOLE', 'applied_transform': src.get('applied_transform'), 'frames': frames}
(S / 'transforms.json').write_text(json.dumps(tj, indent=1)); (S / 'lio_image_poses.json').write_text(json.dumps(lio)); (S / 'lio_image_poses_kf20cm.json').write_text(json.dumps(lio))
print(f'{T}: {len(frames)}/{len(names)} frames posed (their COLMAP), undistorted {W}x{H}, f {newK[0, 0]:.1f} (scaled raw {K[0, 0]:.1f}), c ({newK[0, 2]:.1f}, {newK[1, 2]:.1f})')
