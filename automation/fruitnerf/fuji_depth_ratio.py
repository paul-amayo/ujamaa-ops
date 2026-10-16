"""Along the pixel rays of our camera k: depth of OUR SfM points observed in that image (metres via s) vs the first surface of THEIR cloud on the
same ray (first their-point within 2.5 cm of the ray). Ratio != 1 = the camera sits at the wrong distance from the scene after alignment."""
import struct, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); side = sys.argv[1]; al = sys.argv[2]; ks = [int(x) for x in sys.argv[3].split(',')]; S = ROOT / 'high' / side
def read_images_full(path):
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4); q = np.array(struct.unpack('<dddd', f.read(32))); tt = np.array(struct.unpack('<ddd', f.read(24))); cid = struct.unpack('<I', f.read(4))[0]; nmb = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00': break
                nmb += ch
            np2 = struct.unpack('<Q', f.read(8))[0]; arr = np.frombuffer(f.read(24 * np2), dtype=np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')])); w, x, y, z = q
            Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]); out[nmb.decode()] = (Rq, tt, cid, arr['x'].copy(), arr['y'].copy(), arr['id'].copy())
    return out
def read_points3D_ids(path):
    ids, pts = [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for i in range(n):
            pid = struct.unpack('<Q', f.read(8))[0]; xyz = struct.unpack('<ddd', f.read(24)); f.read(3 + 8); nt = struct.unpack('<Q', f.read(8))[0]; f.read(8 * nt); ids.append(pid); pts.append(xyz)
    return dict(zip(ids, pts))
ims = read_images_full(S / 'sparse/0/images.bin'); P3 = read_points3D_ids(S / 'sparse/0/points3D.bin'); z_ = np.load(S / f'fruit_registry/{al}.npz'); s, Rm, t = float(z_['s']), z_['R'], z_['t']
cloud = np.load(ROOT / 'their_cloud_full_xyz.npy').astype(np.float64); tree = cKDTree(cloud)
for k in ks:
    Rq, tt, cid, xs, ys, pid = ims[f'image_{k}.png']; m = pid >= 0; X = np.array([P3[i] for i in pid[m]]); zc = (Rq @ X.T).T + tt; zo = zc[:, 2] * s   # our depth in metres
    Xg = (s * (Rm @ X.T)).T + t; C = s * (Rm @ (-Rq.T @ tt)) + t; d = Xg - C; d /= np.linalg.norm(d, axis=1)[:, None]
    steps = np.arange(0.3, 12.0, 0.02); Q = C[None, None, :] + d[:, None, :] * steps[None, :, None]; dist, _ = tree.query(Q.reshape(-1, 3), k=1, distance_upper_bound=0.025, workers=8); hit = np.isfinite(dist).reshape(len(d), len(steps))
    first = np.where(hit.any(1), hit.argmax(1), -1); zt = np.where(first >= 0, steps[np.maximum(first, 0)], np.nan); r = zt / zo; ok = np.isfinite(r)
    print(f"[{side} {al} image_{k}] our points {len(zo)} (depth median {np.median(zo):.2f} m, p10 {np.percentile(zo, 10):.2f} p90 {np.percentile(zo, 90):.2f}); their first surface on the ray: found for {100 * ok.mean():.0f} %; ratio their/ours median {np.nanmedian(r):.3f} (p25 {np.nanpercentile(r, 25):.3f} p75 {np.nanpercentile(r, 75):.3f}); |ratio-1| < 0.1 for {100 * np.mean(np.abs(r[ok] - 1) < 0.1):.0f} % of found")
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 20)):
        mm = ok & (zo >= lo) & (zo < hi)
        if mm.sum() > 10: print(f"      our depth {lo}-{hi} m: n {mm.sum()}, their/ours median {np.median(r[mm]):.3f}, their - ours median {np.median(zt[mm] - zo[mm]):+.2f} m")
