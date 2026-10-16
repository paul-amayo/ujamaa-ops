"""fuji_region_check.py <out.png> (2026-10-16) — COLMAP binary readers (read_cameras / read_images / read_points3D, imported by the other fuji_* checks) and, as a script, the top-down map of the dataset cloud with our cameras under align_coarse / align_ground.
Is camera 225 looking at the right region of their cloud? Top-down map (their cloud, its tall columns, our cameras under each alignment,
camera 225 + its view direction, GT apples), the camera-to-canopy distance in both frames, and where their tall columns project in image 225."""
import struct, sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R = ROOT / 'high'
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
def read_points3D(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; pts = np.empty((n, 3))
        for i in range(n):
            f.read(8); pts[i] = struct.unpack('<ddd', f.read(24)); f.read(3 + 8); nt = struct.unpack('<Q', f.read(8))[0]; f.read(8 * nt)
    return pts
if __name__ == '__main__':
    cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); c0 = cloud.mean(0); w_, v_ = np.linalg.eigh(np.cov((cloud[:, :2] - c0[:2]).T)); row2 = v_[:, -1]; acr2 = np.array([-row2[1], row2[0]])
    def rowframe(P): return np.stack([(P[:, :2] - c0[:2]) @ row2, (P[:, :2] - c0[:2]) @ acr2, P[:, 2]], 1)
    rng = np.random.default_rng(0); sub = cloud[rng.choice(len(cloud), 1500000, replace=False)]; subrf = rowframe(sub); ground = np.percentile(cloud[:, 2], 2)
    import glob
    gt_files = sorted(glob.glob('/home/paperspace/data/public/fuji_sfm/Fuji-SfM_dataset/Fuji-SfM_dataset/3-3D_data/GroundTruth-3D_apples_locations/*/*.txt')); GT = np.array([np.loadtxt(f)[0] for f in gt_files]); GTrf = rowframe(GT); print(f"their cloud ground z (p2) {ground:.2f}; GT z range {GT[:, 2].min():.2f}..{GT[:, 2].max():.2f} (above ground {GT[:, 2].min() - ground:.2f}..{GT[:, 2].max() - ground:.2f} m)")
    # tall columns in their cloud: 10 cm xy cells with z extent > 2.5 m
    rf = rowframe(cloud); hi = rf[:, 2] > ground + 2.2; cells = np.floor(rf[hi, :2] / 0.1).astype(int); keys, inv, cnt = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    zmin = np.full(len(keys), np.inf); zmax = np.full(len(keys), -np.inf); np.minimum.at(zmin, inv, rf[hi, 2]); np.maximum.at(zmax, inv, rf[hi, 2])
    tall = (zmax - zmin > 2.0) & (cnt > 150); cols = keys[tall] * 0.1 + 0.05; print(f"their tall columns (cells with > 2 m of points above {ground + 2.2:.1f}): {len(cols)} cells"); 
    for k_ in np.where(tall)[0]: print(f"   along {keys[k_][0] * 0.1 + 0.05:+.2f} across {keys[k_][1] * 0.1 + 0.05:+.2f}  z {zmin[k_]:.2f}..{zmax[k_]:.2f}  n {cnt[k_]}")
    fig, ax = plt.subplots(figsize=(16, 7)); h2 = subrf[subrf[:, 2] > ground + 0.5]; ax.hist2d(h2[:, 0], h2[:, 1], bins=[np.arange(-12, 12, 0.05), np.arange(-6, 6, 0.05)], cmap='Greys', vmax=40)
    ax.scatter(GTrf[:, 0], GTrf[:, 1], s=2, c='red', label='GT apples'); 
    if tall.any(): ax.scatter(cols[:, 0], cols[:, 1], s=120, marker='^', c='blue', label='their tall columns')
    subtree = cKDTree(sub)
    for side, colr in (('east', 'tab:green'), ('west', 'tab:orange')):
        S = R / side; cams = read_cameras(S / 'sparse/0/cameras.bin'); ims = read_images(S / 'sparse/0/images.bin'); X = read_points3D(S / 'sparse/0/points3D.bin'); Xc = X[np.linalg.norm(X - np.median(X, 0), axis=1) < 30]
        for al, mk in (('align_coarse', 'x'), ('align_ground', 'o')):
            z_ = np.load(S / f'fruit_registry/{al}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']
            names = sorted(ims, key=lambda n: int(n.split('_')[1].split('.')[0])); Cs = np.array([s * (Rm @ (-Rq.T @ tt)) + t for n in names for (Rq, tt, cid) in [ims[n]]]); dirs = np.array([Rm @ Rq.T @ np.array([0, 0, 1.0]) for n in names for (Rq, tt, cid) in [ims[n]]])
            Crf = rowframe(Cs); ax.plot(Crf[:, 0], Crf[:, 1], mk, ms=3, c=colr, label=f'{side} cameras ({al})', alpha=0.7)
            k = names.index('image_225.png'); C225 = Cs[k]; d225 = dirs[k]; crf = rowframe(C225[None])[0]; drf = np.array([d225[:2] @ row2, d225[:2] @ acr2, d225[2]])
            ax.annotate(f'{side} 225 ({al.split("_")[1]})', (crf[0], crf[1]), fontsize=8, color=colr); ax.arrow(crf[0], crf[1], 1.5 * drf[0], 1.5 * drf[1], width=0.03, color=colr)
            # camera-to-canopy distance: ours (sparse points above our ground, metres via s) vs theirs (nearest their-cloud point above ground)
            Xg = (s * (Rm @ Xc.T).T) + t; Xg = Xg[Xg[:, 2] > ground + 0.3]; tr_ours = cKDTree(Xg); dn_ours, _ = tr_ours.query(Cs, k=1); subc = sub[sub[:, 2] > ground + 0.3]; dn_theirs, _ = cKDTree(subc).query(Cs, k=1)
            model, W, H, par = cams[ims['image_225.png'][2]]; fx = par[0]; hfov = np.degrees(2 * np.arctan(W / (2 * fx)))
            print(f"== {side} {al}: scale {s:.4f}; cam 225 at along {crf[0]:+.2f} across {crf[1]:+.2f} z {crf[2]:.2f} (above their ground {crf[2] - ground:+.2f} m); view dir along/across/up = {drf[0]:+.2f} {drf[1]:+.2f} {drf[2]:+.2f}; hfov {hfov:.0f} deg")
            print(f"   camera-to-nearest-canopy point, median over cameras: OUR cloud {np.median(dn_ours):.2f} m, THEIR cloud {np.median(dn_theirs):.2f} m; cam 225: ours {dn_ours[k]:.2f}, theirs {dn_theirs[k]:.2f}")
            print(f"   all cameras: along {Crf[:, 0].min():+.1f}..{Crf[:, 0].max():+.1f}, across median {np.median(Crf[:, 1]):+.2f}, z above their ground median {np.median(Crf[:, 2]) - ground:+.2f}")
            # their tall columns through camera 225: pixel positions of column points
            Rq, tt, cid = ims['image_225.png']; Rot = Rq @ Rm.T; cx, cy = par[2], par[3]
            for k_ in np.where(tall)[0]:
                ca, cb = keys[k_] * 0.1 + 0.05; m = (np.abs(rf[:, 0] - ca) < 0.15) & (np.abs(rf[:, 1] - cb) < 0.15) & (rf[:, 2] > ground + 2.2); P = cloud[m]; pc = (1.0 / s) * (Rot @ (P - C225).T).T; zc = pc[:, 2]; ok = zc > 0.1
                if ok.sum() == 0: print(f"   column at along {ca:+.2f}: behind camera"); continue
                u = fx * pc[ok, 0] / zc[ok] + cx; v = fx * pc[ok, 1] / zc[ok] + cy; inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                print(f"   column at along {ca:+.2f} across {cb:+.2f}: {len(P)} pts, range {np.median(zc[ok]) * s:.1f} m, u median {np.median(u):.0f} v {np.median(v):.0f}..{np.max(v):.0f} ({W}x{H}) -> {100 * inside.mean():.0f} % in frame")
    ax.set_xlim(-12, 12); ax.set_ylim(-6, 6); ax.set_aspect('equal'); ax.set_xlabel('along the row [m]'); ax.set_ylabel('across the row [m]'); ax.legend(fontsize=7, loc='upper right'); fig.tight_layout(); out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('fuji_region_map.png'); fig.savefig(out, dpi=90); print('wrote', out)
