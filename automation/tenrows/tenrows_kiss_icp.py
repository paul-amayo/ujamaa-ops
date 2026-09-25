"""LiDAR-only odometry (KISS-ICP) over the dumped ten_rows laser scans, no IMU, no prior. Writes
<out>/lo_poses.npz {ts_ms [N], T [N,4,4] sensor->world, first = identity} and prints the run summary.
  /home/paperspace/envs/lo/bin/python tenrows_kiss_icp.py [--dump <dir>] [--voxel 0.4] [--min_range 0.6] [--max_range 60]"""
import argparse, time, numpy as np
from pathlib import Path
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
ap = argparse.ArgumentParser(); ap.add_argument("--dump", default="/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/laser_dump")
ap.add_argument("--voxel", type=float, default=0.4); ap.add_argument("--min_range", type=float, default=0.6); ap.add_argument("--max_range", type=float, default=60.0)
ap.add_argument("--max_points", type=int, default=0, help="random subsample per scan (0 = all)"); ap.add_argument("--out", default="")
a = ap.parse_args(); D = Path(a.dump); OUT = Path(a.out) if a.out else D
n = int(np.load(D / "n_scans.npy")[0]); ts = np.load(D / "scan_ts_ms.npy")[:n]; scans = np.load(D / "scans_f32.npy", mmap_mode="r")
cfg = KISSConfig(); cfg.mapping.voxel_size = a.voxel; cfg.data.min_range = a.min_range; cfg.data.max_range = a.max_range; cfg.data.deskew = False
odom = KissICP(cfg); T = np.zeros((n, 4, 4)); t0 = time.time(); rng = np.random.default_rng(0)
print(f"[kiss] {n} scans; voxel {a.voxel} m, range {a.min_range}-{a.max_range} m, deskew off; kiss-icp config: adaptive threshold {cfg.adaptive_threshold.initial_threshold}", flush=True)
for i in range(n):
    pts = np.asarray(scans[i], np.float64); pts = pts[np.abs(pts).sum(1) > 0]
    if a.max_points and len(pts) > a.max_points: pts = pts[rng.choice(len(pts), a.max_points, replace=False)]
    odom.register_frame(pts, np.zeros(len(pts)))
    T[i] = np.asarray(odom.last_pose) if hasattr(odom, "last_pose") else np.asarray(odom.poses[-1])
    if i % 250 == 0 or i == n - 1: print(f"[kiss] {i + 1}/{n} ({time.time() - t0:.0f}s, {(i + 1) / max(time.time() - t0, 1e-6):.1f} Hz) pos {np.round(T[i][:3, 3], 2).tolist()}", flush=True)
np.savez(OUT / "lo_poses.npz", ts_ms=ts, T=T)
P = T[:, :3, 3]; print(f"[kiss] done in {(time.time() - t0) / 60:.1f} min: path {np.linalg.norm(np.diff(P, axis=0), axis=1).sum():.1f} m, footprint {np.ptp(P, 0).round(1).tolist()} -> {OUT / 'lo_poses.npz'}", flush=True)
