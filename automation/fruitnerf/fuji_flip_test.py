"""fuji_flip_test.py (2026-10-16, alignment of record = fruit_registry/align_flip.npz) — the face-swap test that fixed the Fuji registration:
every similarity ICP of a one-sided SfM cloud onto the dataset's two-faced row slab converged to the MIRRORED face (east cameras west of the row);
the fix is a 180-degree turn about the vertical axis + the same full-cloud ICP. Prints which side of the row (easting) the cameras sit on under
align_ground and after the flip, cloud inliers, apples-near-GT (a readout only, never fitted) and the ground profile; writes align_flip.npz per side.
system python3; needs fruit_registry/align_ground.npz (the pre-flip fit) and masklink_components_plain30k_record.npz per side.

Face-swap test: which side of the row (easting) do our cameras sit on under an alignment, and does a 180-degree turn about the vertical axis
(= viewing the row from the other face, row direction reversed) followed by the same full-cloud ICP fit better and put the apples on the GT?"""
import sys, json, struct, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R = ROOT / 'high'
from fuji_region_check import read_cameras, read_images, read_points3D   # COLMAP binary readers
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); S_ = np.eye(3); S_[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt)); Rm = U @ S_ @ Vt; s = np.trace(np.diag(D) @ S_) / (Xc ** 2).sum() * len(X); return s, Rm, my - s * Rm @ mx
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); c0 = cloud.mean(0); w_, v_ = np.linalg.eigh(np.cov((cloud[:, :2] - c0[:2]).T)); row2 = v_[:, -1]; acr2 = np.array([-row2[1], row2[0]]); ground = np.percentile(cloud[:, 2], 2)
print(f"their frame: row axis (x,y) = ({row2[0]:+.3f}, {row2[1]:+.3f}), across axis = ({acr2[0]:+.3f}, {acr2[1]:+.3f}); x = easting if UTM: across>0 is {'EAST' if acr2[0] > 0 else 'WEST'} of the row")
rng = np.random.default_rng(0); sub = cloud[rng.choice(len(cloud), 1500000, replace=False)]; sub_tree = cKDTree(sub)
gt_files = sorted(glob.glob(str(ROOT / 'Fuji-SfM_dataset/Fuji-SfM_dataset/3-3D_data/GroundTruth-3D_apples_locations/*/*.txt'))); GT = np.array([np.loadtxt(f)[0] for f in gt_files]); gtree = cKDTree(GT)
def icp_all(X, s, Rm, t, schedule=(0.5, 0.3, 0.2, 0.12, 0.08, 0.05)):
    for dmax in schedule:
        for it in range(6):
            Y = (s * (Rm @ X.T)).T + t; d, j = sub_tree.query(Y, k=1); m = d <= dmax
            if m.sum() < 2000: break
            s, Rm, t = umeyama(X[m], sub[j[m]])
    Y = (s * (Rm @ X.T)).T + t; d, _ = sub_tree.query(Y, k=1); return s, Rm, t, d
def ground_profile(P, step=1.0):
    t_ = (P[:, :2] - c0[:2]) @ row2; a_ = (P[:, :2] - c0[:2]) @ acr2; ys, zs = [], []
    for y0 in np.arange(-7, 7.01, step):
        m = (t_ >= y0) & (t_ < y0 + step) & (np.abs(a_) < 2.5)
        if m.sum() > 200: ys.append(y0 + step / 2); zs.append(np.percentile(P[m, 2], 2))
    return np.array(ys), np.array(zs)
yt, zt = ground_profile(cloud)
for side in ('east', 'west'):
    S = R / side; X = read_points3D(S / 'sparse/0/points3D.bin'); X = X[np.linalg.norm(X - X.mean(0), axis=1) < np.percentile(np.linalg.norm(X - X.mean(0), axis=1), 97)]; ims = read_images(S / 'sparse/0/images.bin'); Cs_our = np.array([-Rq.T @ tt for (Rq, tt, cid) in ims.values()])
    dpt = json.loads((S / 'outputs/sf_30000_bilateral/splatfacto/run/dataparser_transforms.json').read_text()); T = np.array(dpt['transform']); sc = float(dpt['scale']); d_ = np.load(S / 'fruit_registry/masklink_components_plain30k_record.npz'); A2 = (((d_['centroid'][d_['apple']]) / sc) - T[:, 3]) @ T[:, :3]
    z = np.load(S / 'fruit_registry/align_ground.npz'); s0, R0, t0 = float(z['s']), z['R'], z['t']
    results = {}
    for name, (s, Rm, t) in (('ground', (s0, R0, t0)),):
        Cg = (s * (Rm @ Cs_our.T)).T + t; print(f"== {side} align_ground: cameras at across {np.median((Cg[:, :2] - c0[:2]) @ acr2):+.2f} m, easting - cloud mean {np.median(Cg[:, 0]) - c0[0]:+.2f} m -> cameras are {'EAST' if np.median(Cg[:, 0]) > c0[0] else 'WEST'} of the row")
        Y = (s * (Rm @ X.T)).T + t; d, _ = sub_tree.query(Y, k=1); Ag = (s * (Rm @ A2.T)).T + t; dg, _ = gtree.query(Ag, k=1); results[name] = (s, Rm, t); print(f"   cloud inliers 5 / 10 cm: {100 * np.mean(d <= 0.05):.0f} / {100 * np.mean(d <= 0.10):.0f} %; apples within 5 / 10 / 15 cm of GT: {100 * np.mean(dg <= 0.05):.0f} / {100 * np.mean(dg <= 0.10):.0f} / {100 * np.mean(dg <= 0.15):.0f} %")
    # 180 deg about the vertical axis through the row centre at the cameras' mean along-row position
    Cg = (s0 * (R0 @ Cs_our.T)).T + t0; p0 = c0.copy(); p0[:2] = c0[:2] + row2 * np.mean((Cg[:, :2] - c0[:2]) @ row2); Rz = np.diag([-1.0, -1.0, 1.0]); R1 = Rz @ R0; t1 = Rz @ (t0 - p0) + p0; s1 = s0
    s2, R2, t2, d = icp_all(X, s1, R1, t1); Cg2 = (s2 * (R2 @ Cs_our.T)).T + t2; Ag = (s2 * (R2 @ A2.T)).T + t2; dg, _ = gtree.query(Ag, k=1); yo, zo = ground_profile((s2 * (R2 @ X.T)).T + t2)
    print(f"== {side} FLIPPED (180 deg about vertical) + full-cloud ICP: scale {s2:.4f}; cameras at across {np.median((Cg2[:, :2] - c0[:2]) @ acr2):+.2f} m -> {'EAST' if np.median(Cg2[:, 0]) > c0[0] else 'WEST'} of the row; cloud inliers 5 / 10 cm: {100 * np.mean(d <= 0.05):.0f} / {100 * np.mean(d <= 0.10):.0f} %; apples within 5 / 10 / 15 cm of GT: {100 * np.mean(dg <= 0.05):.0f} / {100 * np.mean(dg <= 0.10):.0f} / {100 * np.mean(dg <= 0.15):.0f} %")
    print(f"   ground diff their - ours per bin: " + ' '.join(f"{zt[yt == y][0] - zo[yo == y][0]:+.2f}" for y in np.intersect1d(yt, yo)))
    np.savez(S / 'fruit_registry/align_flip.npz', s=s2, R=R2, t=t2)
