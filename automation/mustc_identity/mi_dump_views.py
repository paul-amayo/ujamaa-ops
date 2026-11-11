#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3, 2026-11-11), step 1: the views of one rig stop in the geometry the HiGH v3 tools rasterise.

splat_census.py / splat_feature_render.py take each frame's camera from the checkpoint's datasets WITHOUT loading the cached
images, so they rasterise a pinhole camera with the workspace intrinsics as written (fl_x, fl_y, cx, cy of transforms_ba.json)
and ignore the OPENCV distortion terms. The rig JPEGs are distorted (k1 about -0.009). This script reads the same cameras the
same way and resamples every raw JPEG into exactly that pinhole (cv2.undistort with the new camera matrix equal to K, same size),
so a mask drawn on the dumped view lines up with what the tools rasterise. Writes <out>/views/<label>.png and <out>/cameras.json
(K, size, camera-to-world in model units, split, dataparser transform and scale, distortion as read).
nerf_new pixi env:  mi_dump_views.py --config <run config.yml> --out <dir>"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from nerfstudio.utils.eval_utils import eval_setup
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
OUT = Path(a.out); (OUT / 'views').mkdir(parents=True, exist_ok=True)
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); dm = pipe.datamanager
dp = json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))
rec = {'config': a.config, 'step': int(step), 'dataparser_transform': dp['transform'], 'dataparser_scale': float(dp['scale']), 'frames': {}}
for split, ds in (('train', dm.train_dataset), ('eval', dm.eval_dataset)):
    for i, p in enumerate(ds.image_filenames):
        cam = ds.cameras[i:i + 1]; K = cam.get_intrinsics_matrices()[0].double().numpy(); W, H = int(cam.width.item()), int(cam.height.item())
        d = cam.distortion_params[0].double().numpy() if cam.distortion_params is not None else np.zeros(6)   # nerfstudio order k1 k2 k3 k4 p1 p2
        dist = np.array([d[0], d[1], d[4], d[5], d[2]])                                                        # OpenCV order k1 k2 p1 p2 k3
        im = cv2.imread(str(p), cv2.IMREAD_COLOR); assert im is not None and im.shape[1] == W and im.shape[0] == H, (p, None if im is None else im.shape, W, H)
        und = cv2.undistort(im, K, dist, None, K)
        name = Path(p).name; label = Path(p).stem
        cv2.imwrite(str(OUT / 'views' / f'{label}.png'), und)
        # displacement of the resampling at the image corner, for the record
        pts = np.array([[[0.0, 0.0]], [[W - 1.0, H - 1.0]], [[W / 2.0, H / 2.0]]]); up = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
        rec['frames'][name] = {'label': label, 'split': split, 'K': K.tolist(), 'w': W, 'h': H, 'c2w_model': cam.camera_to_worlds[0].double().numpy().tolist(),
                               'distortion_ns_k1k2k3k4p1p2': d.tolist(), 'undistort_shift_px_corner00_cornerWH_centre': np.linalg.norm(up - pts.reshape(-1, 2), axis=1).round(3).tolist()}
        print(f'[dump] {name} ({split}): {W}x{H}, fx {K[0, 0]:.1f}, corner shift {rec["frames"][name]["undistort_shift_px_corner00_cornerWH_centre"]} px', flush=True)
json.dump(rec, open(OUT / 'cameras.json', 'w'), indent=1)
print(f'[dump] {len(rec["frames"])} views -> {OUT}/views; cameras.json written', flush=True)
