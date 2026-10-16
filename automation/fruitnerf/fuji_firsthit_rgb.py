"""Proper point render of THEIR coloured cloud from our camera k (align file): colour of the first surface along each ray (no blur). photo | render."""
import sys, struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); side, al, k, step, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; S = ROOT / 'high' / side
from fuji_region_check import read_cameras, read_images, read_points3D   # COLMAP binary readers
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); rgb = np.load(ROOT / 'their_cloud_full_rgb.npy'); tree = cKDTree(cloud)
z_ = np.load(S / f'fruit_registry/{al}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; ims = read_images(S / 'sparse/0/images.bin'); cams = read_cameras(S / 'sparse/0/cameras.bin'); Rq, tt, cid = ims[f'image_{k}.png']; model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; C = s * (Rm @ (-Rq.T @ tt)) + t; Rot = Rq @ Rm.T
vv, uu = np.mgrid[step // 2:H:step, step // 2:W:step]; sh = uu.shape; rc = np.stack([(uu.ravel() - cx) / fx, (vv.ravel() - cy) / fy, np.ones(uu.size)], 1); rc /= np.linalg.norm(rc, axis=1)[:, None]; rays = (Rot.T @ rc.T).T; steps = np.arange(0.3, 10.0, 0.025)
img = np.full((len(rays), 3), 255, np.uint8)
for i0 in range(0, len(rays), 20000):
    Q = C[None, None, :] + rays[i0:i0 + 20000, None, :] * steps[None, :, None]; d, j = tree.query(Q.reshape(-1, 3), k=1, distance_upper_bound=0.03, workers=8); h = np.isfinite(d).reshape(-1, len(steps)); first = h.argmax(1); has = h.any(1); jj = j.reshape(-1, len(steps))[np.arange(len(first)), first]; img[i0:i0 + 20000][has] = rgb[jj[has]]
img = img.reshape(sh + (3,)); photo = np.asarray(Image.open(S / 'images' / f'image_{k}.png').convert('RGB')); ph = np.asarray(Image.fromarray(photo).resize((sh[1], sh[0]))); Image.fromarray(np.concatenate([ph, img], 1)).save(out); print('wrote', out, sh)
