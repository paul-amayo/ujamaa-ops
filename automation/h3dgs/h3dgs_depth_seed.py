#!/usr/bin/env python
"""Dense seed for an H3DGS chunk from the SfM-scaled DA-V2 depth maps (for surveys without usable LiDAR inits):
back-project every `stride`-th pixel of each chunk camera's inverse-depth map (scaled with the chunk's per-image
depth_params, the same values the depth loss uses), keep points inside the cell + margin, subsample to --max, and
write sparse/0/points3D.ply (original kept as points3D_colmap.ply).
  python h3dgs_depth_seed.py <proj_dir> <chunk> [--stride 4] [--max 2500000] [--margin 5] [--every 1]
"""
import argparse, json, shutil, sys, numpy as np, cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, qvec2rotmat
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("chunk"); ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--max", type=int, default=2_500_000); ap.add_argument("--margin", type=float, default=5.0); ap.add_argument("--every", type=int, default=1, help="use every n-th camera")
ap.add_argument("--max_depth", type=float, default=40.0)
a = ap.parse_args(); P = Path(a.proj); cd = P / "camera_calibration/chunks" / a.chunk; CC = P / "camera_calibration"
cams, ims, _ = read_model(str(cd / "sparse/0"), ".bin"); cam = cams[1]; fx, fy, cx, cy = cam.params[:4]
dp = json.load(open(cd / "sparse/0/depth_params.json")); c = np.loadtxt(cd / "center.txt"); e = np.loadtxt(cd / "extent.txt")
lo, hi = c[:2] - e[:2] / 2 - a.margin, c[:2] + e[:2] / 2 + a.margin
pts, cols, used = [], [], 0
for k, im in enumerate(sorted(ims.values(), key=lambda i: i.name)):
    if k % a.every: continue
    stem = im.name.rsplit(".", 1)[0]; prm = dp.get(stem)
    if not prm or prm["scale"] <= 0: continue
    inv = cv2.imread(str(CC / "rectified/depths" / (stem + ".png")), cv2.IMREAD_UNCHANGED)
    if inv is None: continue
    if inv.ndim != 2: inv = inv[..., 0]
    inv = inv.astype(np.float32) / (2 ** 16) * prm["scale"] + prm["offset"]
    rgb = cv2.imread(str(CC / "rectified/images" / im.name))[..., ::-1]
    H, W = inv.shape; sy = rgb.shape[0] / H; sx = rgb.shape[1] / W
    v, u = np.mgrid[0:H:a.stride, 0:W:a.stride]; iv = inv[v, u]; ok = iv > 1.0 / a.max_depth
    z = 1.0 / iv[ok]; uu = u[ok] * sx; vv = v[ok] * sy
    x = (uu - cx) / fx * z; y = (vv - cy) / fy * z; Pc = np.stack([x, y, z], 1)
    R = qvec2rotmat(im.qvec); t = im.tvec; Pw = (R.T @ (Pc - t).T).T   # w2c -> c2w
    m = (Pw[:, 0] >= lo[0]) & (Pw[:, 0] <= hi[0]) & (Pw[:, 1] >= lo[1]) & (Pw[:, 1] <= hi[1])
    pts.append(Pw[m]); cols.append(rgb[vv[m].astype(int).clip(0, rgb.shape[0] - 1), uu[m].astype(int).clip(0, rgb.shape[1] - 1)]); used += 1
xyz = np.concatenate(pts); rgb = np.concatenate(cols).astype(np.uint8)
print(f"[depth-seed] chunk {a.chunk}: {len(xyz)} points from {used} cameras (stride {a.stride}, every {a.every}); cell+{a.margin:g} m")
if len(xyz) > a.max:
    sel = np.random.default_rng(0).choice(len(xyz), a.max, replace=False); xyz, rgb = xyz[sel], rgb[sel]; print(f"[depth-seed] subsampled to {a.max}")
ply = cd / "sparse/0/points3D.ply"; bak = cd / "sparse/0/points3D_colmap.ply"
if not bak.exists(): shutil.copy(ply, bak)
arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
arr["x"], arr["y"], arr["z"] = xyz.T; arr["nx"] = arr["ny"] = arr["nz"] = 0; arr["red"], arr["green"], arr["blue"] = rgb.T
PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply)); print(f"[depth-seed] wrote {ply} ({len(arr)} points)")
