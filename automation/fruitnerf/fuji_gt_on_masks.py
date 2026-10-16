"""GT apple centres projected into our image k (align file), occlusion-tested against THEIR cloud (first hit within 15 % of the GT depth):
fraction landing within one mask radius of one of our SAM3 apple-mask centres; figure = photo with masks (green) and visible GT (yellow)."""
import sys, struct, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from PIL import Image, ImageDraw
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); side, al, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else None; S = ROOT / 'high' / side
from fuji_region_check import read_cameras, read_images, read_points3D   # COLMAP binary readers
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); tree = cKDTree(cloud)
gt_files = sorted(glob.glob(str(ROOT / 'Fuji-SfM_dataset/Fuji-SfM_dataset/3-3D_data/GroundTruth-3D_apples_locations/*/*.txt'))); GT = np.array([np.loadtxt(f)[0] for f in gt_files])
z_ = np.load(S / f'fruit_registry/{al}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; ims = read_images(S / 'sparse/0/images.bin'); cams = read_cameras(S / 'sparse/0/cameras.bin'); Rq, tt, cid = ims[f'image_{k}.png']; model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]; C = s * (Rm @ (-Rq.T @ tt)) + t; Rot = Rq @ Rm.T
pc = (1.0 / s) * (Rot @ (GT - C).T).T; z = pc[:, 2]; okz = z > 0.5; xn, yn = pc[okz, 0] / z[okz], pc[okz, 1] / z[okz]; r2 = xn ** 2 + yn ** 2; rad = 1 + k1 * r2 + k2 * r2 ** 2   # OPENCV distortion -> raw-image pixels (masks are on the raw photo)
xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2); yd = yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn; u = fx * xd + cx; v = fy * yd + cy; inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
idx = np.where(okz)[0][inside]; u, v = u[inside], v[inside]; d = GT[idx] - C; dist = np.linalg.norm(d, axis=1); rays = d / dist[:, None]; steps = np.arange(0.3, 10.0, 0.02)
Q = C[None, None, :] + rays[:, None, :] * steps[None, :, None]; dd, _ = tree.query(Q.reshape(-1, 3), k=1, distance_upper_bound=0.03, workers=8); h = np.isfinite(dd).reshape(len(rays), len(steps)); first = np.where(h.any(1), steps[h.argmax(1)], np.nan); vis = np.isfinite(first) & (first >= 0.85 * dist)
mk = np.asarray(Image.open(S / 'fruit_masks' / f'image_{k}.png')); vals, cnts = np.unique(mk, return_counts=True); fruit = mk == vals[np.argmin(cnts)]; lab, n = ndimage.label(fruit); cents = np.array(ndimage.center_of_mass(fruit, lab, range(1, n + 1))); areas = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); keep = areas >= 30; cents, areas = cents[keep], areas[keep]; radii = np.sqrt(areas / np.pi)
ct = cKDTree(cents[:, ::-1]); dm, jm = ct.query(np.stack([u[vis], v[vis]], 1), k=1); on = dm <= radii[jm]; on2 = dm <= 2 * radii[jm]
print(f"[{side} {al} image_{k}] GT in frame {len(u)}, visible by their cloud {vis.sum()}; our masks {len(cents)}; visible GT within 1 mask radius of a mask centre {100 * on.mean():.0f} %, within 2 radii {100 * on2.mean():.0f} %; masks claimed by a visible GT {len(np.unique(jm[on]))} / {len(cents)}; median GT-to-nearest-mask-centre {np.median(dm):.0f} px (mask radius median {np.median(radii):.0f} px)")
if out:
    im = Image.open(S / 'images' / f'image_{k}.png').convert('RGB'); dr = ImageDraw.Draw(im)
    for (cy_, cx_), r_ in zip(cents, radii): dr.ellipse([cx_ - r_, cy_ - r_, cx_ + r_, cy_ + r_], outline=(0, 255, 0), width=2)
    for uu, vv, v_ in zip(u, v, vis): dr.ellipse([uu - 6, vv - 6, uu + 6, vv + 6], outline=(255, 255, 0) if v_ else (255, 0, 255), width=3)
    im.save(out); print('wrote', out)
