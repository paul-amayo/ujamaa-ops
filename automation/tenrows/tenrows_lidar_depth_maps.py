"""LiDAR depth maps for an H3DGS project of ten_rows (Paul, 2026-09-26: "add the lidar to this h3dgs survey, then we
have a comparison"). For every image in <proj>/camera_calibration/rectified/images (kf_K.png and kf_K_R.png): the laser
scans within +-max_dt of the keyframe stamp (laser_dump/scans_f32.npy, raw sensor frame) are motion-compensated with the
KISS-ICP poses to the keyframe stamp, moved into the camera with the rig extrinsic (left: L2C; right: T_rig @ L2C),
projected with the project camera and z-buffered into a 16-bit inverse-depth PNG (value = invdepth[1/m] / S * 65536,
S = --inv_scale; 0 = no return) in <proj>/camera_calibration/rectified/depths_lidar/, plus depth_params_lidar.json
{name: {scale: S, offset: 0, med_scale: S}} so H3DGS's cameras.py turns the PNG back into metric inverse depth.
Splat radius --radius px (z-buffer keeps the nearest). Run with the h3dgs env (numpy, PIL)."""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R_ / "prod/monos/monolithics"; DUMP = R_ / "experimental/laser_dump"
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--max_dt", type=float, default=60.0); ap.add_argument("--radius", type=int, default=1)
ap.add_argument("--inv_scale", type=float, default=2.0); ap.add_argument("--min_range", type=float, default=0.45); ap.add_argument("--max_range", type=float, default=40.0); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); P = Path(a.proj); IM = P / "camera_calibration/rectified/images"; OUT = P / "camera_calibration/rectified/depths_lidar"; OUT.mkdir(parents=True, exist_ok=True)
cam = json.load(open(P / "export_meta.json"))["camera"]; fx, fy, cx, cy, W, H = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["w"], cam["h"]
L2C = np.array(json.load(open(R_ / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64)
T_rig = np.array(json.load(open(R_ / "experimental/stereo_rig_left_to_right.json"))["T_rig_left_to_right"], np.float64)
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(MD / "kf_index.json"))}
z = np.load(DUMP / "lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]; n = int(np.load(DUMP / "n_scans.npy")[0]); scans = np.load(DUMP / "scans_f32.npy", mmap_mode="r")
slerp = Slerp(ts, Rotation.from_matrix(T[:, :3, :3]))
def pose_at(t):
    t = float(np.clip(t, ts[0], ts[-1])); i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1); w = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
    M = np.eye(4); M[:3, :3] = slerp([t]).as_matrix()[0]; M[:3, 3] = (1 - w) * T[i - 1, :3, 3] + w * T[i, :3, 3]; return M
names = sorted(p.name for p in IM.glob("kf_*.png")); names = names[: a.limit] if a.limit else names
params = {}; cover = []; t0 = time.time(); yy, xx = np.mgrid[-a.radius:a.radius + 1, -a.radius:a.radius + 1]; disc = (yy ** 2 + xx ** 2) <= a.radius ** 2; oy, ox = yy[disc], xx[disc]
for j, name in enumerate(names):
    K = int(re.search(r"kf_(\d+)", name).group(1)); right = name.endswith("_R.png"); tk = kf[K]; Twl_k = pose_at(tk); E = (T_rig @ L2C) if right else L2C
    idx = np.where(np.abs(ts - tk) <= a.max_dt)[0]; idx = idx[idx < n]
    inv = np.zeros((H, W), np.float32); zbuf = np.full((H, W), np.inf, np.float32)
    for i in idx:
        Pts = np.asarray(scans[i], np.float64); Pts = Pts[np.abs(Pts).sum(1) > 0]; r = np.linalg.norm(Pts, axis=1); Pts = Pts[(r >= a.min_range) & (r <= a.max_range)]
        M = E @ np.linalg.inv(Twl_k) @ T[i]                       # scan sensor frame at its stamp -> world -> laser at the keyframe stamp -> camera
        Pc = Pts @ M[:3, :3].T + M[:3, 3]; f = Pc[:, 2] > 0.2; Pc = Pc[f]
        u = np.round(fx * Pc[:, 0] / Pc[:, 2] + cx).astype(int); v = np.round(fy * Pc[:, 1] / Pc[:, 2] + cy).astype(int); d = Pc[:, 2].astype(np.float32)
        for dy, dx in zip(oy, ox):
            uu, vv = u + dx, v + dy; m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv, dd = uu[m], vv[m], d[m]; order = np.argsort(-dd); uu, vv, dd = uu[order], vv[order], dd[order]   # nearest written last
            zbuf[vv, uu] = np.minimum(zbuf[vv, uu], dd)
    valid = np.isfinite(zbuf); inv[valid] = 1.0 / zbuf[valid]
    png = np.clip(inv / a.inv_scale * 65536.0, 0, 65535).astype(np.uint16); Image.fromarray(png).save(OUT / name)
    params[name[:-4]] = {"scale": a.inv_scale, "offset": 0.0, "med_scale": a.inv_scale}; cover.append(valid.mean())
    if j % 200 == 0: print(f"[lidar-depth] {j}/{len(names)} {name}: {len(idx)} scans, coverage {valid.mean():.1%}, depth median {np.median(zbuf[valid]) if valid.any() else float('nan'):.2f} m ({time.time() - t0:.0f}s)", flush=True)
json.dump(params, open(P / "camera_calibration/rectified/depth_params_lidar.json", "w"))
print(f"[lidar-depth] {len(names)} maps -> {OUT}; coverage median {np.median(cover):.1%} p10 {np.percentile(cover, 10):.1%}; depth_params_lidar.json (scale {a.inv_scale}, offset 0) in {(time.time() - t0) / 60:.1f} min", flush=True)
