#!/usr/bin/env python3
"""fuji_cloud_splat_render.py <side> <k> [--sigma 0.02] [--n 2000000] — render the dataset's coloured dense cloud DENSELY through our camera
by splatting every point as a small isotropic gaussian with gsplat (the point render covered 31 % of the pixels; MI needs a full image).
Pose = our COLMAP pose through the per-side similarity (or --pose-file mi_pose_<k>.npz). The photo is undistorted with our OPENCV
coefficients so both images share the pinhole model. Writes fruit_registry/cloud_splat_<k>.png (photo | splat render). nerf_new pixi env."""
import argparse, struct
from pathlib import Path
import numpy as np, cv2, torch
from gsplat import rasterization
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('side'); ap.add_argument('k', type=int); ap.add_argument('--sigma', type=float, default=0.02); ap.add_argument('--n', type=int, default=2000000); ap.add_argument('--pose-file', default=None); ap.add_argument('--align', default='align_coarse'); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); S = ROOT / 'high' / a.side; nm = f'image_{a.k}.png'
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); rgb = np.load(ROOT / 'their_cloud_full_rgb.npy')
def read_cameras(path):
    cams = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack('<IiQQ', f.read(24)); npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}[model]; cams[cid] = (model, w, h, np.array(struct.unpack('<' + 'd' * npar, f.read(8 * npar))))
    return cams
def read_images(path):
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4); q = np.array(struct.unpack('<dddd', f.read(32))); tt = np.array(struct.unpack('<ddd', f.read(24))); cid = struct.unpack('<I', f.read(4))[0]; nmb = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00': break
                nmb += ch
            np2 = struct.unpack('<Q', f.read(8))[0]; f.read(24 * np2); w, x, y, z = q; Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]); out[nmb.decode()] = (Rq, tt, cid)
    return out
cams = read_cameras(S / 'sparse/0/cameras.bin'); ims = read_images(S / 'sparse/0/images.bin'); Rq, tt, cid = ims[nm]; model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]
z_ = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']
if a.pose_file: pz = np.load(S / 'fruit_registry' / a.pose_file); Rot, C = pz['R'], pz['C']
else: Rot, C = Rq @ Rm.T, s * (Rm @ (-Rq.T @ tt)) + t
# cloud in the camera frame (metric = their units / s to match our camera's COLMAP units); subsample near points
pc = (1.0 / s) * (Rot @ (cloud - C).T).T; z = pc[:, 2]; ok = (z > 0.3) & (z < 10.0 / s); idx = np.where(ok)[0]; rng = np.random.default_rng(0)
if len(idx) > a.n: idx = idx[rng.choice(len(idx), a.n, replace=False)]
P = torch.tensor(pc[idx], dtype=torch.float32, device='cuda'); col = torch.tensor(rgb[idx] / 255.0, dtype=torch.float32, device='cuda'); N = len(idx)
viewmat = torch.eye(4, device='cuda')[None]; K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device='cuda')[None]
sig = a.sigma / s; scales = torch.full((N, 3), sig, device='cuda'); quats = torch.zeros((N, 4), device='cuda'); quats[:, 0] = 1; opac = torch.full((N,), 0.9, device='cuda')
out, alpha, _ = rasterization(means=P, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=viewmat, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e3, render_mode='RGB')
img = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8); cov = (alpha[0, ..., 0].cpu().numpy() > 0.5).mean()
photo = np.array(Image.open(S / 'images' / nm).convert('RGB')); Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); photo_u = cv2.undistort(photo, Kc, np.array([k1, k2, p1, p2]))
cv2.imwrite(str(S / 'fruit_registry' / f'cloud_splat_{a.k}.png'), cv2.cvtColor(np.concatenate([photo_u, img], 1), cv2.COLOR_RGB2BGR)); np.save(S / 'fruit_registry' / f'cloud_splat_{a.k}.npy', img)
print(f'[splat-render] {nm}: {N} gaussians (sigma {a.sigma * 100:.0f} cm), coverage (alpha > 0.5) {100 * cov:.0f} % -> cloud_splat_{a.k}.png')
