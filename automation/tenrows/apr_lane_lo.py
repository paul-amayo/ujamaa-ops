"""April Klapmuts lane 2, December-style LiDAR odometry base (UJAMAA 2026-10-06). The first April base (apr_lane_base.py:
export keyframe poses slerped to the stream, scans placed through them) measured 2.5x less self-consistent than December's
LO placement (scan-to-scan ICP correction at the camera, frames 2-10 m apart: median 10.2 cm vs 4.3 cm, growing with the
separation), so the lane gets what December's lane recipe had (tenrows_lane_prep.py lo + tenrows_lo_lidar_init.py --stamps):
  dump  the lane window's laser scans (+-3 s) from laser.monolithic -> <lane>/laser_dump/{scans_f32,scan_ts_ms,n_scans}.npy
        (tenrows_laser_dump.py format), then KISS-ICP over them with December's settings (tenrows_kiss_icp.py, lo env)
  base  every stream frame: LO pose slerped at the frame stamp x laser->camera^-1, OpenGL c2w, ZED-conf camera
        -> <lane>/transforms_lo.json
  lidar every frame's nearest scan (<= 60 ms) placed with its LO pose, range 0.45-40 m, lane camera box + (6, 6, 4) m,
        5 cm voxels, coloured from the frame (ZED-conf projection) -> <lane>/init_lidar.ply
  window December's lane-window rule (tenrows_lane_windows.py) on the lane's LO track: the straight core (moving > 0.3 m/s
        along the row axis, within 0.6 m laterally of the middle-half median) + 1.5 s pads -> <lane>/lane_window.json; base
        then keeps only the frames inside it (the April keyframe window kf 193-348 starts in the ~100 deg turn into the lane,
        where the warp's position-based yaw went wrong by up to 7.6 deg; December's lanes leave the turns out by this rule;
        PAD_START_S=0 drops the start pad: April's turn-in pad is a ~40 deg pivot in 1.5 s, the frames the warp got wrong)
usage: apr_lane_lo.py dump <lane>   (nerf_new python3.10 + aru_py_logger)   |   apr_lane_lo.py window|base <lane>   (h3dgs env)"""
import json, re, sys
from pathlib import Path
import numpy as np
mode, LD = sys.argv[1], Path(sys.argv[2]); M = Path("/home/paperspace/data/klapmuts/apr_2026_zed/prod/monos"); D = LD / "laser_dump"
FX, FY, CX, CY, W, H = 527.985, 527.88, 638.975, 333.1835, 1280, 720   # the ZED conf December's lane recipe used (same sensor head)
stamps = {k: float(v) for k, v in json.load(open(LD / "stamps.json")).items()}; names = sorted(stamps, key=lambda n: int(re.sub(r"\D", "", n)))
if mode == "base" and (LD / "lane_window.json").exists():
    _w = json.load(open(LD / "lane_window.json")); names = [n for n in names if _w["t0_ms"] <= stamps[n] <= _w["t1_ms"]]
if mode == "dump":
    sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
    t_lo, t_hi = min(stamps.values()) - 3000, max(stamps.values()) + 3000; D.mkdir(exist_ok=True); lg = aru_py_logger.LaserLogger(str(M / "laser.monolithic"), False); S, TS = [], []
    while True:   # LaserLogger has no end_of_file()
        try: r = lg.read_from_file()
        except Exception: break
        if r is None or r[0] is None or getattr(r[0], "size", 0) == 0: break
        t = int(r[1])
        if t < t_lo: continue
        if t > t_hi: break
        p = np.zeros((64000, 3), np.float32); q = np.asarray(r[0], np.float32)[:, :3]; p[:len(q)] = q[:64000]; S.append(p); TS.append(t)
    np.save(D / "scans_f32.npy", np.stack(S)); np.save(D / "scan_ts_ms.npy", np.array(TS, np.int64)); np.save(D / "n_scans.npy", np.array([len(S)]))
    print(f"[apr-lo] dumped {len(S)} scans {TS[0]}..{TS[-1]} ms ({(TS[-1] - TS[0]) / 1000:.1f} s; frames span {(max(stamps.values()) - min(stamps.values())) / 1000:.1f} s) -> {D}", flush=True)
elif mode == "window":   # tenrows_lane_windows.py on this lane's LO (horizontal plane = LO x, y; the dump is one lane, so the plane SVD of a near-straight track is not used)
    z = np.load(D / "lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); P = z["T"][:, :3, 3]; xy = P[:, :2] - P[:, :2].mean(0)
    dt = np.gradient(ts) / 1e3; vel = np.gradient(xy, axis=0) / dt[:, None]; speed = np.linalg.norm(vel, axis=1); moving = speed > 0.3
    ang = np.arctan2(vel[moving, 1], vel[moving, 0]); axis_ang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
    e1 = np.array([np.cos(axis_ang), np.sin(axis_ang)]); e2 = np.array([-e1[1], e1[0]]); s_ = xy @ e1; l = xy @ e2
    win = max(3, int(round(1.0 / np.median(dt)))); vs = np.convolve(vel @ e1, np.ones(win) / win, mode="same")
    legs = np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)]); m = max((np.nonzero(legs == L_)[0] for L_ in np.unique(legs)), key=lambda m_: np.ptp(s_[m_]))
    mid = m[len(m) // 4: 3 * len(m) // 4]; lat0 = np.median(l[mid]); core = m[(np.abs(vs[m]) > 0.3) & (np.abs(l[m] - lat0) < 0.6)]
    pad0 = float(__import__("os").environ.get("PAD_START_S", "1.5")); i0, i1 = core.min(), core.max(); t0, t1 = ts[i0] - pad0 * 1e3, ts[i1] + 1.5e3; inside = [n for n in names if t0 <= stamps[n] <= t1]
    json.dump(dict(t0_ms=int(t0), t1_ms=int(t1), seconds=round((t1 - t0) / 1e3, 1), length_m=round(float(abs(s_[i1] - s_[i0])), 1), rule=f"tenrows_lane_windows.py (straight core + pads {pad0:g} s start / 1.5 s end) on the lane LO",
                   first=inside[0], last=inside[-1], n_frames=len(inside)), open(LD / "lane_window.json", "w"), indent=1)
    print(f"[apr-lo] lane window {(t1 - t0) / 1e3:.1f} s, {abs(s_[i1] - s_[i0]):.1f} m core along the axis: {len(inside)}/{len(names)} frames {inside[0]}..{inside[-1]} (left out: {names.index(inside[0])} at the start, {len(names) - 1 - names.index(inside[-1])} at the end)", flush=True)
elif mode == "base":
    import cv2
    from scipy.spatial.transform import Rotation, Slerp
    z = np.load(D / "lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]; slerp = Slerp(ts, Rotation.from_matrix(T[:, :3, :3])); GL = np.diag([1.0, -1.0, -1.0, 1.0])
    L2C = np.array(json.load(open(M / "rig.json"))["laser_to_camera_left"], np.float64); C2L = np.linalg.inv(L2C); frames = []; C2W = {}
    for n in names:   # tenrows_lane_prep.py lo, verbatim construction
        t = float(np.clip(stamps[n], ts[0], ts[-1])); i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)); w = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
        Mx = np.eye(4); Mx[:3, :3] = slerp([t]).as_matrix()[0]; Mx[:3, 3] = (1 - w) * T[i - 1, :3, 3] + w * T[i, :3, 3]; C2W[n] = Mx @ C2L
        frames.append({"file_path": str(LD / "images" / n), "transform_matrix": (C2W[n] @ GL).tolist(), "timestamp_ns": int(stamps[n])})
    J = {"fl_x": FX, "fl_y": FY, "cx": CX, "cy": CY, "w": W, "h": H, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV", "ply_file_path": "init_lidar.ply",
         "pose_convention": "opengl_c2w (KISS-ICP LiDAR odometry over the lane's scans, slerp'd to the frame stamp x laser->camera extrinsic; metric)", "frames": frames}
    (LD / "transforms_lo.json").write_text(json.dumps(J, indent=1)); C = np.array([C2W[n][:3, 3] for n in names])
    print(f"[apr-lo] transforms_lo.json: {len(frames)} frames, path {np.linalg.norm(np.diff(C, axis=0), axis=1).sum():.1f} m, median step {np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)) * 100:.1f} cm", flush=True)
    scans = np.load(D / "scans_f32.npy", mmap_mode="r"); n_s = int(np.load(D / "n_scans.npy")[0]); lo, hi = C.min(0) - [6, 6, 4], C.max(0) + [6, 6, 4]; P, RGB, used = [], [], 0
    for n in names:   # tenrows_lo_lidar_init.py --stamps: every frame's nearest scan, placed with its own LO pose
        j = int(np.argmin(np.abs(ts - stamps[n])))
        if abs(ts[j] - stamps[n]) > 60: continue
        p = np.asarray(scans[j], np.float64); p = p[np.abs(p).sum(1) > 0]; r = np.linalg.norm(p, axis=1); p = p[(r > 0.45) & (r < 40)]
        pw = (T[j][:3, :3] @ p.T).T + T[j][:3, 3]; w2c = np.linalg.inv(C2W[n]); pc = (w2c[:3, :3] @ pw.T).T + w2c[:3, 3]; z_ = pc[:, 2]
        u = np.round(FX * pc[:, 0] / np.maximum(z_, 1e-6) + CX).astype(int); v = np.round(FY * pc[:, 1] / np.maximum(z_, 1e-6) + CY).astype(int)
        img = cv2.imread(str(LD / "images" / n))[:, :, ::-1]; col = np.full((len(pw), 3), 128, np.uint8); ok = (z_ > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H); col[ok] = img[v[ok], u[ok]]
        keep = np.all((pw >= lo) & (pw <= hi), axis=1); P.append(pw[keep]); RGB.append(col[keep]); used += 1
    P = np.concatenate(P); RGB = np.concatenate(RGB); _, first = np.unique(np.floor(P / 0.05).astype(np.int64), axis=0, return_index=True); P, RGB = P[first], RGB[first]
    with open(LD / "init_lidar.ply", "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        rec = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]); rec["x"], rec["y"], rec["z"] = P.T; rec["red"], rec["green"], rec["blue"] = RGB.T; f.write(rec.tobytes())
    print(f"[apr-lo] init_lidar.ply: {len(P):,} points (5 cm voxels) from {used}/{len(names)} frames' nearest scans ({n_s} scans), {100 * float((RGB != 128).any(1).mean()):.0f} % coloured", flush=True)
