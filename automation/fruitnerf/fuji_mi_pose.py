#!/usr/bin/env python3
"""fuji_mi_pose.py <side> <image index> [--res 3] — register one of our photos to the dataset's COLOURED dense cloud by MUTUAL INFORMATION
(2026-10-16, Paul: "the cloud has colour, project it to an image, there is mutual information"). The cloud is z-buffer-rendered through our
calibrated camera from a 6-DoF pose (start = our COLMAP pose through the per-side similarity); holes are filled by a small dilation and the
render is blurred; the objective is MI(photo intensity, render intensity) over covered pixels plus MI(photo hue-ish redness, render redness)
— maximised with Powell from several perturbed starts. Validation that uses the ground truth only as a probe: fraction of GT apple centres
that land within one mask radius of a SAM3 mask after the pose. Writes fruit_registry/mi_pose_<K>.npz (R, C in the cloud frame) and an overlay."""
import argparse, json, struct
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pycocotools import mask as mask_utils
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('side'); ap.add_argument('k', type=int); ap.add_argument('--res', type=int, default=3); ap.add_argument('--restarts', type=int, default=6); ap.add_argument('--align', default='align_coarse'); a = ap.parse_args()
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
z_ = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; Rot0 = Rq @ Rm.T; C0 = s * (Rm @ (-Rq.T @ tt)) + t   # their frame; our camera units = their / s
f_ = a.res; w2, h2 = W // f_, H // f_; Kmat = np.array([[fx / f_, 0, cx / f_], [0, fy / f_, cy / f_], [0, 0, 1]]); dist = np.array([k1, k2, p1, p2])
photo = cv2.resize(np.array(Image.open(S / 'images' / nm).convert('RGB')), (w2, h2), interpolation=cv2.INTER_AREA).astype(np.float32)
pg = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY); pr = photo[..., 0] - 0.5 * (photo[..., 1] + photo[..., 2])   # redness
# cloud subset near the start pose (12 m) for speed
near = np.where(np.linalg.norm(cloud - C0, axis=1) < 9.0)[0]; rng0 = np.random.default_rng(0); near = near[rng0.choice(len(near), min(400000, len(near)), replace=False)]; CL = cloud[near]; CR = rgb[near].astype(np.float32); print(f'[mi] {nm}: {len(CL)} cloud points within 12 m; render {w2}x{h2}', flush=True)
def render(Rot, C):
    pc = (1.0 / s) * (Rot @ (CL - C).T).T; z = pc[:, 2]; ok = (z > 0.3) & (z < 9.0 / s)
    if ok.sum() < 100: return None, None
    pts, _ = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kmat, dist); uv = pts.reshape(-1, 2); zz = z[ok]; col = CR[ok]
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int); ins = (u >= 0) & (u < w2) & (v >= 0) & (v < h2); u, v, zz, col = u[ins], v[ins], zz[ins], col[ins]; order = np.argsort(-zz)
    img = np.zeros((h2, w2, 3), np.float32); cov = np.zeros((h2, w2), bool); img[v[order], u[order]] = col[order]; cov[v[order], u[order]] = True
    img = cv2.dilate(img, np.ones((3, 3), np.uint8)); cov = cv2.dilate(cov.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); img = cv2.GaussianBlur(img, (5, 5), 0); return img, cov
def mi(x, y, bins=32):
    h, _, _ = np.histogram2d(x, y, bins=bins); p = h / h.sum(); px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0; return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
def objective(p, base_R, base_C):
    Rot = Rotation.from_rotvec(p[:3]).as_matrix() @ base_R; C = base_C + p[3:]; img, cov = render(Rot, C)
    if img is None or cov.sum() < 2000: return 1.0
    rg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); rr = img[..., 0] - 0.5 * (img[..., 1] + img[..., 2]); m = cov
    if cov.mean() < 0.6 * COV0: return 1.0   # the pose may not escape to a view with much less of the cloud in it
    return -(mi(pg[m], rg[m]) + mi(pr[m], rr[m]))
# GT probe
sam = S / 'prod/bateleur/sam3_v2'; ents = None
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, e in meta['frame_entries'].items():
        if f"image_{l2i[int(li)]}.png" == nm: ents = sorted(e, key=lambda e: e.get('score', 0))
cent = np.array([np.argwhere(mask_utils.decode(e['rle'])).mean(0)[::-1] for e in ents]); rad = np.array([np.sqrt(mask_utils.area(e['rle']) / np.pi) for e in ents]); ctree = cKDTree(cent); Kfull = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
def gt_probe(Rot, C):
    pc = (1.0 / s) * (Rot @ (GT - C).T).T; z = pc[:, 2]; ok = z > 0.3; pts, _ = cv2.projectPoints(pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kfull, dist); uv = pts.reshape(-1, 2); ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if ins.sum() == 0: return 0, 0.0, 0.0
    d, j = ctree.query(uv[ins], k=1); return int(ins.sum()), 100 * np.mean(d <= rad[j]), 100 * np.mean(d <= 2 * rad[j])
_img0, _cov0 = render(Rot0, C0); COV0 = _cov0.mean(); print(f'[mi] start coverage {100 * COV0:.0f} %', flush=True)
print('[mi] start: objective %.3f; GT probe: %d in frame, on a mask within 1 / 2 radii %.0f / %.0f %%' % (objective(np.zeros(6), Rot0, C0), *gt_probe(Rot0, C0)), flush=True)
rng = np.random.default_rng(0); best = (objective(np.zeros(6), Rot0, C0), np.zeros(6))
for trial in range(a.restarts):
    x0 = np.zeros(6) if trial == 0 else np.r_[rng.uniform(-0.12, 0.12, 3), rng.uniform(-0.8, 0.8, 3)]
    res = minimize(objective, x0, args=(Rot0, C0), method='Powell', bounds=[(-0.26, 0.26)] * 3 + [(-1.5, 1.5)] * 3, options={'maxiter': 6, 'maxfev': 600, 'xtol': 2e-3, 'ftol': 1e-4})   # rotation <= 15 deg, shift <= 1.5 m per axis
    Rr = Rotation.from_rotvec(res.x[:3]).as_matrix() @ Rot0; Cr = C0 + res.x[3:]
    print('[mi]   restart %d: objective %.3f, moved %.2f m, rotated %.1f deg; GT probe %d in frame, %.0f / %.0f %%' % (trial, res.fun, np.linalg.norm(res.x[3:]), np.degrees(np.linalg.norm(res.x[:3])), *gt_probe(Rr, Cr)), flush=True)
    if res.fun < best[0]: best = (res.fun, res.x)
p = best[1]; Rr = Rotation.from_rotvec(p[:3]).as_matrix() @ Rot0; Cr = C0 + p[3:]; np.savez(S / f'fruit_registry/mi_pose_{a.k}.npz', R=Rr, C=Cr, objective=best[0])
print('[mi] best: objective %.3f (start %.3f), camera moved %.2f m, rotated %.1f deg; GT probe %d in frame, on a mask within 1 / 2 radii %.0f / %.0f %%' % (best[0], objective(np.zeros(6), Rot0, C0), np.linalg.norm(p[3:]), np.degrees(np.linalg.norm(p[:3])), *gt_probe(Rr, Cr)), flush=True)
img0, _ = render(Rot0, C0); img1, _ = render(Rr, Cr); over = np.concatenate([photo, img0, img1], 1).astype(np.uint8); cv2.imwrite(str(S / 'fruit_registry' / f'mi_pose_{a.k}.png'), cv2.cvtColor(over, cv2.COLOR_RGB2BGR)); print(f'[mi] overlay photo | start render | MI render -> mi_pose_{a.k}.png')
