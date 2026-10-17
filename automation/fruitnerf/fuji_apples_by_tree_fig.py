#!/usr/bin/env python3
"""fuji_apples_by_tree_fig.py <side> <k> <out.npz> (2026-10-17) — for the paper's hero figure: our merged Fuji apples (merged_flip_r10.npz, GT frame),
assigned to the trunk anchors of tree_counts_flip_keepposts.json, projected into our image k (align_flip, raw-image distortion) and occlusion-tested
against the dataset cloud (first hit within 15 % of the apple's depth); writes the photo, the apple pixels and their tree index. system python3."""
import sys, json, struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); from fuji_region_check import read_cameras, read_images
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); side, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]; S = ROOT / 'high' / side
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); tree = cKDTree(cloud); c500 = np.load(ROOT / 'their_cloud_500k.npy').astype(np.float64); c0 = c500.mean(0); w_, v_ = np.linalg.eigh(np.cov((c500[:, :2] - c0[:2]).T)); row2 = v_[:, -1]
M = np.load(ROOT / 'merged_flip_r10.npz'); P = M['P']; TR = np.array(json.load(open(ROOT / 'tree_counts_flip_keepposts.json'))['trunks_along']); along = (P[:, :2] - c0[:2]) @ row2; assign = np.argmin(np.abs(along[:, None] - TR[None, :]), axis=1)
z_ = np.load(S / 'fruit_registry/align_flip.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; ims = read_images(S / 'sparse/0/images.bin'); cams = read_cameras(S / 'sparse/0/cameras.bin'); Rq, tt, cid = ims[f'image_{k}.png']; model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]; C = s * (Rm @ (-Rq.T @ tt)) + t; Rot = Rq @ Rm.T
pc = (1.0 / s) * (Rot @ (P - C).T).T; z = pc[:, 2]; ok = z > 0.5; xn, yn = pc[ok, 0] / z[ok], pc[ok, 1] / z[ok]; r2 = xn ** 2 + yn ** 2; rad = 1 + k1 * r2 + k2 * r2 ** 2
u = fx * (xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)) + cx; v = fy * (yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn) + cy; inside = (u >= 0) & (u < W) & (v >= 0) & (v < H); idx = np.where(ok)[0][inside]; u, v = u[inside], v[inside]
d = P[idx] - C; dist = np.linalg.norm(d, axis=1); rays = d / dist[:, None]; steps = np.arange(0.3, 10.0, 0.02); Q = C[None, None, :] + rays[:, None, :] * steps[None, :, None]; dd, _ = tree.query(Q.reshape(-1, 3), k=1, distance_upper_bound=0.03, workers=8)
h = np.isfinite(dd).reshape(len(rays), len(steps)); first = np.where(h.any(1), steps[h.argmax(1)], np.nan); vis = np.isfinite(first) & (first >= 0.85 * dist)
photo = np.asarray(Image.open(S / 'images' / f'image_{k}.png').convert('RGB')); np.savez_compressed(out, photo=photo, u=u[vis], v=v[vis], tree=assign[idx][vis], depth=dist[vis], trunks=TR); print(f"[apples-by-tree] {side} image_{k}: {len(idx)} apples in frame, {int(vis.sum())} visible, trees present {sorted(set(assign[idx][vis].tolist()))} -> {out}")
