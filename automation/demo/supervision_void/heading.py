# 4 heading.py (h3dgs env): camera heading vs the scene-graph row direction (0 deg = looking down the lane)
import json, sys, numpy as np
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, qvec2rotmat
S = '/home/paperspace/data/citrus_all/01_13B_Jackal'; src = f'{S}/experimental/h3dgs/camera_calibration/chunks/3_1/sparse/0'
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}
row_of = {o['id']: o['row_id'] for o in json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
T = {int(k): np.array(v) for k, v in json.load(open('/home/paperspace/logs/demo_chunks/01_3_1_720/demo_path.json'))['trees'].items()}
dirs = []
for r in (16, 17, 18, 19, 20):
    P = np.array([T[t][:2] for t in T if row_of.get(t) == r])
    if len(P) >= 3:
        d = np.linalg.svd(P - P.mean(0))[2][0]; dirs.append(d if (not dirs or d @ dirs[0] > 0) else -d); print(f'row {r}: {len(P)} trees, ground-plane direction {np.round(dirs[-1], 3)}')
rd = np.mean(dirs, 0); rd /= np.linalg.norm(rd)
for k in (3112, 3113, 3115):
    im = ims[f'kf_{k:06d}.png']; R = qvec2rotmat(im.qvec); fwd = R.T[:, 2][:2]; fwd /= np.linalg.norm(fwd)
    print(f'kf_{k}: camera heading vs row direction {np.degrees(np.arccos(abs(fwd @ rd))):.0f} deg (0 = looking down the lane)')
