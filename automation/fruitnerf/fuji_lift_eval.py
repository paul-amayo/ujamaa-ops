#!/usr/bin/env python3
"""fuji_lift_eval.py — evaluate on Fuji-SfM the way its authors did (Gené-Mola et al. 2020): 2-D detections lifted onto THEIR dense cloud,
matched to the GT boxes by volume overlap. For each of our images (pose in their frame = our COLMAP pose through the per-side similarity
align_coarse), their cloud points within the camera's view are projected; the points inside each SAM3 mask are DBSCAN-grouped at 3 cm and
the group nearest the camera is the detection (a 10 cm box at its centre, like the GT boxes); detections from all images and both sides are
merged when their boxes overlap by > 50 %; a merged detection is a true positive when it overlaps a GT box by > 50 %. Reports P / R / F1,
the multi-detection rate, and the count. Our detections thus inherit their cloud's errors exactly as the GT does — no apple-level
registration anywhere. system python3 (numpy/scipy):  fuji_lift_eval.py [--sides east west] [--step 1] [--min-pts 5]"""
import argparse, json, struct, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ap = argparse.ArgumentParser(); ap.add_argument('--sides', nargs='+', default=['east', 'west']); ap.add_argument('--step', type=int, default=1); ap.add_argument('--min-pts', type=int, default=5); ap.add_argument('--align', default='align_coarse'); ap.add_argument('--box', type=float, default=0.10); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); DS = ROOT / 'Fuji-SfM_dataset/Fuji-SfM_dataset'; R = ROOT / 'high'
gt_files = sorted(glob.glob(str(DS / '3-3D_data/GroundTruth-3D_apples_locations/*/*.txt'))); GT = np.array([np.loadtxt(f)[0] for f in gt_files]); GT_tree = np.array([Path(f).name.split('-')[0] for f in gt_files])
cloud_cache = ROOT / 'their_cloud_full_xyz.npy'
if not cloud_cache.exists(): np.save(cloud_cache, np.loadtxt(DS / '3-3D_data/Fuji_apple_trees_point_cloud.txt', usecols=(0, 1, 2), dtype=np.float32))
CLOUD = np.load(cloud_cache).astype(np.float64); print(f'[lift] their cloud {len(CLOUD)} points, GT {len(GT)} boxes', flush=True)
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
            f.read(4); q = np.array(struct.unpack('<dddd', f.read(32))); tt = np.array(struct.unpack('<ddd', f.read(24))); cid = struct.unpack('<I', f.read(4))[0]; nm = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00': break
                nm += ch
            np2 = struct.unpack('<Q', f.read(8))[0]; f.read(24 * np2); w, x, y, z = q; Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]); out[nm.decode()] = (Rq, tt, cid)
    return out
from pycocotools import mask as mask_utils
dets = []   # (centre xyz, side, image)
for side in a.sides:
    S = R / side; z = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z['s']), z['R'], z['t']
    cams = read_cameras(S / 'sparse/0/cameras.bin'); ims = read_images(S / 'sparse/0/images.bin'); sam = S / 'prod/bateleur/sam3_v2'; ents_by = {}
    for cm in sorted(sam.glob('clip_*/frame_entries.json')):
        meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
        for li, e in meta['frame_entries'].items(): ents_by[f"image_{l2i[int(li)]}.png"] = e
    names = sorted(ims, key=lambda n: int(n[6:-4]))[::a.step]; ctree = cKDTree(CLOUD); n_det = 0
    for nm in names:
        if nm not in ents_by: continue
        Rq, tt, cid = ims[nm]; model, W, H, par = cams[cid]; fx, fy, cx, cy = par[:4]; k1, k2, p1, p2 = par[4:8]
        # camera in their frame: x_cam = Rq ( R^T (x_their - t) / s ) + tt ; camera centre their-frame:
        Cc = -Rq.T @ tt; Cw = s * (Rm @ Cc) + t; near = ctree.query_ball_point(Cw, r=6.0 * s)   # points within 6 m of the camera
        if len(near) < 100: continue
        P = CLOUD[near]; Pc = ((P - t) @ Rm) / s; pc = (Rq @ Pc.T).T + tt; zc = pc[:, 2]; ok = zc > 0.3; xn, yn = pc[:, 0] / np.where(ok, zc, 1), pc[:, 1] / np.where(ok, zc, 1); r2 = xn ** 2 + yn ** 2; rad = 1 + k1 * r2 + k2 * r2 ** 2
        u = fx * (xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)) + cx; v = fy * (yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn) + cy; ins = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi_ = u[ins].astype(int), v[ins].astype(int); Pi = P[ins]; zi = zc[ins]
        for e in ents_by[nm]:
            m = mask_utils.decode(e['rle']).astype(bool); sel = m[vi_, ui]
            if sel.sum() < a.min_pts: continue
            Q = Pi[sel]; zq = zi[sel]; tree = cKDTree(Q); prs = tree.query_pairs(0.03, output_type='ndarray')
            if len(prs): G = coo_matrix((np.ones(len(prs)), (prs[:, 0], prs[:, 1])), shape=(len(Q), len(Q))); kq, cq = connected_components(G, directed=False)
            else: kq, cq = len(Q), np.arange(len(Q))
            sizes = np.bincount(cq, minlength=kq); groups = [g for g in range(kq) if sizes[g] >= a.min_pts]
            if not groups: continue
            g = min(groups, key=lambda g: np.median(zq[cq == g]))   # nearest group to the camera
            dets.append(Q[cq == g].mean(0)); n_det += 1
    print(f'[lift] {side}: {n_det} per-image 3-D detections from {len(names)} images', flush=True)
D = np.array(dets); print(f'[lift] {len(D)} per-image detections total', flush=True)
# merge across images / sides: boxes (side a.box) overlapping > 50 % <=> centre separation < box/2 per axis (for equal axis-aligned cubes, overlap fraction = prod(1 - |d_i|/box) > 0.5)
def overlap(c1, c2): d = np.abs(c1 - c2) / a.box; return 0.0 if (d >= 1).any() else float(np.prod(1 - d))
tree = cKDTree(D); prs = tree.query_pairs(a.box * np.sqrt(3), output_type='ndarray'); keep = np.array([overlap(D[i], D[j]) > 0.5 for i, j in prs]) if len(prs) else np.zeros(0, bool)
G = coo_matrix((np.ones(int(keep.sum())), (prs[keep, 0], prs[keep, 1])), shape=(len(D), len(D))); k, comp = connected_components(G, directed=False); size = np.bincount(comp, minlength=k)
M = np.array([D[comp == c].mean(0) for c in range(k)]); views = size
for minv in (1, 2, 3):
    Mm = M[views >= minv]; dd = np.array([[overlap(x, g) for g in GT] for x in Mm]) if len(Mm) else np.zeros((0, len(GT)))
    used_g = np.zeros(len(GT), bool); used_m = np.zeros(len(Mm), bool); tp = 0; md = 0
    for kk in np.argsort(-dd, axis=None):
        i, j = divmod(int(kk), len(GT))
        if dd[i, j] <= 0.5: break
        if used_m[i]: continue
        if used_g[j]: md += 1; used_m[i] = True; continue   # another detection of an already-detected apple (multi-detection)
        used_g[j] = used_m[i] = True; tp += 1
    fp = len(Mm) - tp - md; P_ = tp / max(1, len(Mm)); R_ = tp / len(GT)
    print(f'[lift] merged detections seen in >= {minv} images: {len(Mm)}; TP {tp} (GT box overlap > 50 %), multi-detections {md}, FP {fp}; precision {P_:.3f} recall {R_:.3f} F1 {2 * P_ * R_ / max(1e-9, P_ + R_):.3f}; per tree recall: ' + ', '.join(f'{t_}: {used_g[GT_tree == t_].sum()}/{(GT_tree == t_).sum()}' for t_ in sorted(set(GT_tree))), flush=True)
np.save(ROOT / 'lift_detections.npy', M)
