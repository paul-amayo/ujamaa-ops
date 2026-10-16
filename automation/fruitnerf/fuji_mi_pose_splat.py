#!/usr/bin/env python3
"""fuji_mi_pose_splat.py <side> <k> — register one photo to the dataset's coloured cloud by MUTUAL INFORMATION against a DENSE gsplat render
of the cloud (points as isotropic gaussians), coarse-to-fine: level 8 (sigma 5 cm, wide multi-start search: shift ±2 m, rotation ±25 deg),
level 4 (sigma 3 cm) and level 2 (sigma 2 cm) refinements from the best. Objective = MI(intensity) + MI(redness) over pixels where the render
has alpha > 0.3, with a coverage floor. The GT is used only as a probe (fraction of GT centres within one mask radius of a SAM3 mask).
Writes fruit_registry/mi_splat_pose_<k>.npz (R, C in the cloud frame) and the overlay photo | start | result. nerf_new pixi env (GPU)."""
import argparse, json, struct
from pathlib import Path
import numpy as np, cv2, torch
from gsplat import rasterization
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pycocotools import mask as mask_utils
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('side'); ap.add_argument('k', type=int); ap.add_argument('--n', type=int, default=3000000); ap.add_argument('--starts', type=int, default=24); ap.add_argument('--align', default='align_coarse'); ap.add_argument('--span-t', type=float, default=2.0); ap.add_argument('--span-r', type=float, default=0.44); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); S = ROOT / 'high' / a.side; nm = f'image_{a.k}.png'
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); rgb = np.load(ROOT / 'their_cloud_full_rgb.npy')
GT = np.array([np.loadtxt(f)[0] for f in sorted((ROOT / 'Fuji-SfM_dataset/Fuji-SfM_dataset/3-3D_data/GroundTruth-3D_apples_locations').glob('*/*.txt'))])
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
z_ = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; Rot0 = Rq @ Rm.T; C0 = s * (Rm @ (-Rq.T @ tt)) + t
near = np.where(np.linalg.norm(cloud - C0, axis=1) < 12.0)[0]; rng = np.random.default_rng(0); near = near[rng.choice(len(near), min(a.n, len(near)), replace=False)]
CL = torch.tensor(cloud[near], dtype=torch.float32, device='cuda'); COL = torch.tensor(rgb[near] / 255.0, dtype=torch.float32, device='cuda'); N = len(near)
Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); photo_u = cv2.undistort(np.array(Image.open(S / 'images' / nm).convert('RGB')), Kc, np.array([k1, k2, p1, p2]))
quats = torch.zeros((N, 4), device='cuda'); quats[:, 0] = 1; opac = torch.full((N,), 0.9, device='cuda')
def render(Rot, C, lvl, sigma):
    w2, h2 = W // lvl, H // lvl; K = torch.tensor([[fx / lvl, 0, cx / lvl], [0, fy / lvl, cy / lvl], [0, 0, 1]], dtype=torch.float32, device='cuda')[None]
    Rt = torch.tensor(Rot, dtype=torch.float32, device='cuda'); Ct = torch.tensor(C, dtype=torch.float32, device='cuda'); P = (1.0 / s) * ((CL - Ct) @ Rt.T)
    out, alpha, _ = rasterization(means=P, quats=quats, scales=torch.full((N, 3), sigma / s, device='cuda'), opacities=opac, colors=COL, viewmats=torch.eye(4, device='cuda')[None], Ks=K, width=w2, height=h2, sh_degree=None, packed=False, near_plane=0.05, far_plane=1e3, render_mode='RGB')
    return out[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
def mi(x, y, bins=24):
    h, _, _ = np.histogram2d(x, y, bins=bins); p = h / max(h.sum(), 1); px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0; return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
photos = {}
for lvl in (8, 4, 2):
    ph = cv2.resize(photo_u, (W // lvl, H // lvl), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0; photos[lvl] = (cv2.cvtColor(ph, cv2.COLOR_RGB2GRAY), ph[..., 0] - 0.5 * (ph[..., 1] + ph[..., 2]))
def objective(p, base_R, base_C, lvl, sigma, cov0):
    Rot = Rotation.from_rotvec(p[:3]).as_matrix() @ base_R; C = base_C + p[3:]; img, alpha = render(Rot, C, lvl, sigma); m = (alpha > 0.3) & REGION[lvl]   # MI only inside the canopy region of the start view (dilated): the sky-vs-sky optimum (10-16 evening) is excluded
    if m.mean() < 0.85 * cov0: return 1.0
    pg, pr = photos[lvl]; rg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); rr = img[..., 0] - 0.5 * (img[..., 1] + img[..., 2]); return -(mi(pg[m], rg[m]) + mi(pr[m], rr[m]))
sam = S / 'prod/bateleur/sam3_v2'; ents = None
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, e in meta['frame_entries'].items():
        if f"image_{l2i[int(li)]}.png" == nm: ents = sorted(e, key=lambda e: e.get('score', 0))
cent = np.array([np.argwhere(mask_utils.decode(e['rle'])).mean(0)[::-1] for e in ents]); rad = np.array([np.sqrt(mask_utils.area(e['rle']) / np.pi) for e in ents]); ctree = cKDTree(cent); dist = np.array([k1, k2, p1, p2])
def gt_probe(Rot, C):
    pc = (1.0 / s) * (Rot @ (GT - C).T).T; z = pc[:, 2]; ok = z > 0.3; pts, _ = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, dist); uv = pts.reshape(-1, 2); ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if ins.sum() == 0: return 0, 0.0, 0.0
    d, j = ctree.query(uv[ins], k=1); return int(ins.sum()), 100 * np.mean(d <= rad[j]), 100 * np.mean(d <= 2 * rad[j])
print(f'[mi-splat] {nm}: {N} gaussians; start GT probe: %d in frame, on a mask within 1 / 2 radii %.0f / %.0f %%' % gt_probe(Rot0, C0), flush=True)
REGION = {}
for lvl in (8, 4, 2):
    _, al = render(Rot0, C0, lvl, 0.05); REGION[lvl] = cv2.dilate((al > 0.3).astype(np.uint8), np.ones((2 * (24 // lvl) + 1, 2 * (24 // lvl) + 1), np.uint8)).astype(bool)
R_cur, C_cur = Rot0, C0
for lvl, sigma, span_t, span_r, starts, fev in ((8, 0.05, a.span_t, a.span_r, a.starts, 400), (4, 0.03, min(0.6, a.span_t), min(0.15, a.span_r), 6, 400), (2, 0.02, 0.25, 0.06, 3, 400)):
    _, al0 = render(R_cur, C_cur, lvl, sigma); cov0 = ((al0 > 0.3) & REGION[lvl]).mean(); best = (objective(np.zeros(6), R_cur, C_cur, lvl, sigma, cov0), np.zeros(6))
    for trial in range(starts):
        x0 = np.zeros(6) if trial == 0 else np.r_[rng.uniform(-span_r, span_r, 3), rng.uniform(-span_t, span_t, 3)]
        res = minimize(objective, x0, args=(R_cur, C_cur, lvl, sigma, cov0), method='Powell', bounds=[(-span_r, span_r)] * 3 + [(-span_t, span_t)] * 3, options={'maxiter': 8, 'maxfev': fev, 'xtol': 1e-3, 'ftol': 1e-5})
        if res.fun < best[0]: best = (res.fun, res.x)
    p = best[1]; R_cur = Rotation.from_rotvec(p[:3]).as_matrix() @ R_cur; C_cur = C_cur + p[3:]
    print(f'[mi-splat] level {lvl} (sigma {sigma * 100:.0f} cm, {starts} starts): best MI {-best[0]:.3f}; camera now %.2f m from the alignment pose; GT probe %d in frame, %.0f / %.0f %%' % (np.linalg.norm(C_cur - C0), *gt_probe(R_cur, C_cur)), flush=True)
np.savez(S / f'fruit_registry/mi_splat_pose_{a.k}.npz', R=R_cur, C=C_cur)
img0, _ = render(Rot0, C0, 2, 0.02); img1, _ = render(R_cur, C_cur, 2, 0.02); ph = cv2.resize(photo_u, (W // 2, H // 2)); over = np.concatenate([ph, (img0 * 255).astype(np.uint8), (img1 * 255).astype(np.uint8)], 1); cv2.imwrite(str(S / 'fruit_registry' / f'mi_splat_{a.k}.png'), cv2.cvtColor(over, cv2.COLOR_RGB2BGR)); print(f'[mi-splat] overlay -> mi_splat_{a.k}.png', flush=True)
