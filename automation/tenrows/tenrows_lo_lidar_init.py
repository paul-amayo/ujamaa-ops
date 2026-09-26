"""LiDAR init for a ten_rows block from the KISS-ICP odometry (2026-09-26): every keyframe's nearest laser scan (raw
sensor frame, laser_dump/scans_f32.npy) is placed with its LO pose (lo_poses.npz, sensor->world), cropped to the block's
camera bounding box (+pad), voxel-downsampled and written as <block_dir>/init_lidar.ply (x y z red green blue; colour =
the keyframe pixel the point projects to when inside the image, else mid-grey), and <block_dir>/transforms.json's
ply_file_path is set to it. The block's transforms.json must be the LO-world poses (transforms_lo.json or the refined
transforms_ref_lo.json copied there).
  python (h3dgs env: numpy, plyfile, PIL) tenrows_lo_lidar_init.py <block_dir> [--voxel 0.05] [--pad-x 10 --pad-y 8 --pad-z 5] [--min-range 0.45] [--max-range 40]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R_ / "prod/monos/monolithics"; DUMP = R_ / "experimental/laser_dump"
ap = argparse.ArgumentParser(); ap.add_argument("block_dir"); ap.add_argument("--voxel", type=float, default=0.05); ap.add_argument("--pad-x", type=float, default=10.0)
ap.add_argument("--pad-y", type=float, default=8.0); ap.add_argument("--pad-z", type=float, default=5.0); ap.add_argument("--min-range", type=float, default=0.45); ap.add_argument("--max-range", type=float, default=40.0)
ap.add_argument("--max-dt", type=float, default=60.0, help="ms: keyframe-to-scan stamp tolerance"); ap.add_argument("--out-name", default="init_lidar.ply")
ap.add_argument("--stamps", default="", help="json {image name: ts_ms} for frames that are not survey keyframes (e.g. a lane's full-stream frames)")
a = ap.parse_args(); BD = Path(a.block_dir)
J = json.load(open(BD / "transforms.json")); fx, fy, cx, cy, W, H = J["fl_x"], J["fl_y"], J["cx"], J["cy"], J["w"], J["h"]
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(MD / "kf_index.json"))}
STAMPS = {k: float(v) for k, v in json.load(open(a.stamps)).items()} if a.stamps else None
z = np.load(DUMP / "lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]; n = int(np.load(DUMP / "n_scans.npy")[0]); scans = np.load(DUMP / "scans_f32.npy", mmap_mode="r")
GL = np.diag([1.0, -1.0, -1.0, 1.0]); L2C = np.array(json.load(open(R_ / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64)
cams = []
for f in J["frames"]:
    name = Path(f["file_path"]).name
    if STAMPS is not None: k, t = name, STAMPS[name]
    else: k = int(re.search(r"kf_(\d+)", f["file_path"]).group(1)); t = kf[k]
    c2w_cv = np.asarray(f["transform_matrix"], np.float64) @ GL; cams.append((k, t, c2w_cv, f["file_path"]))
C = np.array([c[2][:3, 3] for c in cams]); lo, hi = C.min(0) - [a.pad_x, a.pad_y, a.pad_z], C.max(0) + [a.pad_x, a.pad_y, a.pad_z]
pts, cols, used = [], [], 0
for k, t, c2w_cv, fp in cams:
    i = int(np.argmin(np.abs(ts - t)))
    if abs(ts[i] - t) > a.max_dt or i >= n: continue
    P = np.asarray(scans[i], np.float64); P = P[np.abs(P).sum(1) > 0]; r = np.linalg.norm(P, axis=1); P = P[(r >= a.min_range) & (r <= a.max_range)]
    Pw = P @ T[i][:3, :3].T + T[i][:3, 3]; m = np.all((Pw >= lo) & (Pw <= hi), axis=1); Pw = Pw[m]; P = P[m]
    if not len(Pw): continue
    # colour from the keyframe: laser -> camera (rig extrinsic) -> pixel; points behind or outside stay grey
    Pc = P @ L2C[:3, :3].T + L2C[:3, 3]; col = np.full((len(Pw), 3), 128, np.uint8); vis = Pc[:, 2] > 0.2
    u = (fx * Pc[vis, 0] / Pc[vis, 2] + cx).round().astype(int); v = (fy * Pc[vis, 1] / Pc[vis, 2] + cy).round().astype(int); inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.any():
        try:
            img = np.asarray(Image.open(fp).convert("RGB")); idx = np.where(vis)[0][inside]; col[idx] = img[v[inside], u[inside]]
        except Exception as e: pass
    pts.append(Pw.astype(np.float32)); cols.append(col); used += 1
xyz = np.concatenate(pts); rgb = np.concatenate(cols)
# voxel downsample (first point per voxel)
key = np.floor((xyz - xyz.min(0)) / a.voxel).astype(np.int64); _, first = np.unique(key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791, return_index=True); xyz, rgb = xyz[first], rgb[first]
arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]; arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(BD / a.out_name))
J["ply_file_path"] = a.out_name; (BD / "transforms.json").write_text(json.dumps(J, indent=1))
print(f"[lo-init] {BD.name}: {used}/{len(cams)} keyframes lifted, {len(xyz)} points after {a.voxel} m voxels (coloured {(rgb != 128).any(1).mean():.0%}) -> {BD / a.out_name}", flush=True)
