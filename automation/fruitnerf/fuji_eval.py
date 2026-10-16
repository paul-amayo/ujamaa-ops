#!/usr/bin/env python3
"""fuji_eval.py — per-apple evaluation on Fuji-SfM (2026-10-16): align our SfM frame to the dataset's georeferenced point cloud and match
registry apples to the 1,455 ground-truth apple centres (3-D boxes, 10 cm cubes, row 0 = centre).
  1. our COLMAP sparse points (sparse/0/points3D.bin) vs their cloud (subsampled): coarse init by PCA row axis + camera-up + extent scale,
     all sign combinations tried; trimmed-ICP with a similarity (Umeyama) refines; the init with the lowest inlier RMS wins.
  2. registry apples (model frame of the splat) -> COLMAP frame via dataparser_transforms.json -> GT frame via the similarity.
  3. matching inside the GT region (GT bounding box + margin): greedy nearest within --radius (sweep printed); per tree (GT file names)
     and overall precision / recall / F1. Sides (east, west) are evaluated alone and merged (apples closer than --radius across sides = one).
system python3 (numpy/scipy):  fuji_eval.py --sides east west --dump plain30k_record [--radius 0.15]"""
import argparse, json, struct, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('--sides', nargs='+', default=['east', 'west']); ap.add_argument('--dump', default='plain30k_record'); ap.add_argument('--radius', type=float, default=0.15); ap.add_argument('--margin', type=float, default=0.3); ap.add_argument('--merge', default='one2one', choices=['one2one', 'union-find'], help='cross-side merge: greedy one-to-one east<->west pairs within --radius (default) or the old transitive union-find over every pair'); ap.add_argument('--save-merged', default=None, help='write npz: P (merged apples, GT frame), pair (GT index matched at --radius or -1), GT, GT_tree'); ap.add_argument('--align-only', action='store_true'); ap.add_argument('--align-file', default=None, help='use fruit_registry/<name>.npz (s, R, t) of each side instead of the ICP alignment'); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); DS = ROOT / 'Fuji-SfM_dataset/Fuji-SfM_dataset'; R = ROOT / 'high'
# ---- ground truth ----
gt_files = sorted(glob.glob(str(DS / '3-3D_data/GroundTruth-3D_apples_locations/*/*.txt'))); GT = np.array([np.loadtxt(f)[0] for f in gt_files]); GT_tree = np.array([Path(f).name.split('-')[0] for f in gt_files])
print(f'[fuji] GT apples {len(GT)} in {len(set(GT_tree))} trees; region x {GT[:, 0].min():.1f}..{GT[:, 0].max():.1f} y {GT[:, 1].min():.1f}..{GT[:, 1].max():.1f} z {GT[:, 2].min():.1f}..{GT[:, 2].max():.1f}', flush=True)
cloud_cache = ROOT / 'their_cloud_500k.npy'
if not cloud_cache.exists():
    pts = np.loadtxt(DS / '3-3D_data/Fuji_apple_trees_point_cloud.txt', usecols=(0, 1, 2), dtype=np.float32); rng = np.random.default_rng(0); pts = pts[rng.choice(len(pts), 500000, replace=False)]; np.save(cloud_cache, pts)
THEIR = np.load(cloud_cache).astype(np.float64); their_tree = cKDTree(THEIR)
def read_points3D(path):
    pts = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack('<Q', f.read(8))[0]; xyz = struct.unpack('<ddd', f.read(24)); f.read(3); err = struct.unpack('<d', f.read(8))[0]; tl = struct.unpack('<Q', f.read(8))[0]; f.read(8 * tl); pts.append(xyz)
    return np.array(pts)
def read_images_bin_centers_up(path):
    C, UPs = [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack('<I', f.read(4))[0]; q = np.array(struct.unpack('<dddd', f.read(32))); t = np.array(struct.unpack('<ddd', f.read(24))); cam = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00': break
                name += ch
            np2 = struct.unpack('<Q', f.read(8))[0]; f.read(24 * np2)
            w, x, y, z = q; Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            C.append(-Rm.T @ t); UPs.append(-Rm.T @ np.array([0, 1, 0]))   # camera up = -y of the COLMAP camera
    return np.array(C), np.array(UPs)
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); d = np.ones(3); d[2] = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag(d) @ Vt
    s = (S * d).sum() / (Xc ** 2).sum(1).mean(); t = my - s * Rm @ mx; return s, Rm, t
def icp(X, s, Rm, t, iters=40):
    for it in range(iters):
        Y = (s * (Rm @ X.T)).T + t; d, j = their_tree.query(Y, k=1); dmax = max(0.08, np.percentile(d, 60) if it < 10 else np.percentile(d, 50)); m = d <= dmax
        if m.sum() < 100: return s, Rm, t, np.inf, 0
        s, Rm, t = umeyama(X[m], THEIR[j[m]])
    Y = (s * (Rm @ X.T)).T + t; d, _ = their_tree.query(Y, k=1); inl = d <= 0.1; return s, Rm, t, float(np.sqrt((d[inl] ** 2).mean())) if inl.any() else np.inf, float(inl.mean())
def align(side):
    S = R / side; X = read_points3D(S / 'sparse/0/points3D.bin'); C, UPs = read_images_bin_centers_up(S / 'sparse/0/images.bin')
    # robust subset: points within 3 x the median camera-to-point distance (drop far outliers)
    X = X[np.linalg.norm(X - X.mean(0), axis=1) < np.percentile(np.linalg.norm(X - X.mean(0), axis=1), 95)]
    up = UPs.mean(0); up /= np.linalg.norm(up); Xc = X - X.mean(0); H = Xc - np.outer(Xc @ up, up); w_, v_ = np.linalg.eigh(np.cov(H.T)); row = v_[:, -1]; row -= (row @ up) * up; row /= np.linalg.norm(row); acr = np.cross(up, row)
    Tc = THEIR - THEIR.mean(0); upT = np.array([0, 0, 1.0]); HT = Tc - np.outer(Tc @ upT, upT); wT, vT = np.linalg.eigh(np.cov(HT.T)); rowT = vT[:, -1]; rowT -= (rowT @ upT) * upT; rowT /= np.linalg.norm(rowT); acrT = np.cross(upT, rowT)
    ext_ours = np.percentile(Xc @ row, 97) - np.percentile(Xc @ row, 3); ext_theirs = np.percentile(Tc @ rowT, 97) - np.percentile(Tc @ rowT, 3); s0 = ext_theirs / ext_ours
    best = None
    for sr in (1, -1):
        for sa in (1, -1):
            B_ours = np.stack([row * sr, acr * sa * sr, up], 1); B_theirs = np.stack([rowT, acrT, upT], 1); Rm = B_theirs @ B_ours.T
            if np.linalg.det(Rm) < 0: continue
            t = THEIR.mean(0) - s0 * Rm @ X.mean(0); s, Rm2, t2, rms, frac = icp(X, s0, Rm, t)
            print(f'[fuji] {side}: init row sign {sr:+d} across sign {sa:+d}: scale {s:.3f}, inlier RMS {rms:.3f} m, inliers {100 * frac:.0f} %', flush=True)
            if best is None or (frac > best[4] + 0.05) or (abs(frac - best[4]) <= 0.05 and rms < best[3]): best = (s, Rm2, t2, rms, frac)
    s, Rm, t, rms, frac = best; print(f'[fuji] {side}: alignment scale {s:.4f} (1 unit = {s:.3f} m), inlier RMS {rms * 100:.1f} cm, inliers {100 * frac:.0f} %', flush=True); return s, Rm, t
def apples_in_gt_frame(side, s, Rm, t):
    S = R / side; dump = np.load(S / f'fruit_registry/masklink_components_{a.dump}.npz'); cen = dump['centroid'][dump['apple']]
    dpt = json.loads((S / 'outputs/sf_30000_bilateral/splatfacto/run/dataparser_transforms.json').read_text()); T = np.array(dpt['transform']); sc = float(dpt['scale'])
    colmap = ((cen / sc) - T[:, 3]) @ T[:, :3]   # model = scale * (T @ [p;1]); T[:, :3] is orthonormal -> inverse by transpose
    return (s * (Rm @ colmap.T)).T + t
lo, hi = GT.min(0) - a.margin, GT.max(0) + a.margin; allP = []
for side in a.sides:
    if a.align_file:
        z_ = np.load(R / side / 'fruit_registry' / f'{a.align_file}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']; print(f'[fuji] {side}: alignment from {a.align_file}.npz (scale {s:.4f})', flush=True)
    else: s, Rm, t = align(side)
    if a.align_only: continue
    P = apples_in_gt_frame(side, s, Rm, t); inside = np.all((P >= lo) & (P <= hi), 1); print(f'[fuji] {side}: registry apples {len(P)}, inside the GT region {int(inside.sum())}', flush=True); allP.append(P[inside])
if a.align_only: raise SystemExit(0)
P = np.concatenate(allP)
if len(a.sides) > 1 and a.merge == 'one2one':   # merge the sides: greedy one-to-one east<->west pairs within the radius are one apple (no within-side merging, no chains)
    A_, B_ = allP[0], np.concatenate(allP[1:]); d = np.linalg.norm(A_[:, None, :] - B_[None, :, :], axis=-1); ua = np.zeros(len(A_), bool); ub = np.zeros(len(B_), bool); merged = []
    for k in np.argsort(d, axis=None):
        i, j = divmod(int(k), len(B_))
        if d[i, j] > a.radius: break
        if ua[i] or ub[j]: continue
        ua[i] = ub[j] = True; merged.append((A_[i] + B_[j]) / 2)
    P = np.concatenate([np.array(merged).reshape(-1, 3), A_[~ua], B_[~ub]]); print(f'[fuji] merged sides (one-to-one within {a.radius:.2f} m): {len(merged)} pairs + {int((~ua).sum())} + {int((~ub).sum())} singles = {len(P)} apples', flush=True)
elif len(a.sides) > 1:   # old rule: apples closer than the radius are one (transitive)
    tree = cKDTree(P); pairs = tree.query_pairs(a.radius); par = np.arange(len(P))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i_, j_ in pairs: par[f(i_)] = f(j_)
    roots = np.array([f(i_) for i_ in range(len(P))]); P = np.array([P[roots == r_].mean(0) for r_ in np.unique(roots)]); print(f'[fuji] merged sides: {len(P)} apples', flush=True)
PAIR = None
def match(P, radius):
    global PAIR
    d = np.linalg.norm(P[:, None, :] - GT[None, :, :], axis=-1); used_p = np.zeros(len(P), bool); used_g = np.zeros(len(GT), bool); tp = 0; pair = np.full(len(P), -1)
    for k in np.argsort(d, axis=None):
        i, j = divmod(int(k), len(GT))
        if d[i, j] > radius: break
        if used_p[i] or used_g[j]: continue
        used_p[i] = used_g[j] = True; tp += 1; pair[i] = j
    PAIR = pair; return tp, used_g
for radius in (0.1, 0.15, 0.2, 0.3):
    tp, ug = match(P, radius); prec = tp / max(1, len(P)); rec = tp / len(GT); print(f'[fuji] radius {radius:.2f} m: TP {tp}, FP {len(P) - tp}, FN {len(GT) - tp}; precision {prec:.3f} recall {rec:.3f} F1 {2 * prec * rec / max(1e-9, prec + rec):.3f}', flush=True)
tp, ug = match(P, a.radius)
if a.save_merged: np.savez(a.save_merged, P=P, pair=PAIR, GT=GT, GT_tree=GT_tree, radius=a.radius); print(f'[fuji] merged apples + matching at {a.radius:.2f} m -> {a.save_merged}', flush=True)
print('[fuji] recall per tree at the chosen radius: ' + ', '.join(f'{t_}: {ug[GT_tree == t_].sum()}/{(GT_tree == t_).sum()}' for t_ in sorted(set(GT_tree))), flush=True)
