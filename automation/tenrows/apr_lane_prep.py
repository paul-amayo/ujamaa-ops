"""APRIL copy (2026-10-06, apr_lane_run.sh; survey root apr_2026_zed, same ZED-conf camera as December - the records say the same sensor head; only the chunk mode is used, the lo base comes from apr_lane_base.py). Single-lane H3DGS build for ten_rows (Paul, 2026-09-26: "prove in one row and then scale to the rest"). Two modes:
  lo    <lane dir>                : transforms_lo.json for the lane's full-stream frames (KISS-ICP laser pose slerp'd at
                                    each stamp x L2C^-1, OpenGL c2w, ZED-conf intrinsics) — the base the lane's own SfM
                                    is placed on (tenrows_lane_warp.py / klapmuts_apply_refine_orient.py)
  chunk <lane dir> <ref transforms> [--proj h3dgs] [--keyframes]
                                  : the H3DGS project layout under <lane dir>/<proj>: camera_calibration/chunks/lane/
                                    sparse/0 (COLMAP bin model in the LO world, one PINHOLE camera; LiDAR init as
                                    points3D.ply; test.txt = every 10th frame of the FULL stream, identical for every
                                    project of the lane), center/extent = lane bounding box, rectified/images (hardlinks),
                                    aligned/sparse/0 (cameras + test.txt) and a minimal export_meta.json so the chunk
                                    evaluator runs. --keyframes (ablation, Paul 2026-09-27 "is it more frames?") keeps
                                    only the training frames that are >= 20 cm or >= 3 deg from the last kept one (the
                                    survey's keyframe rule); the held-out frames stay in the model for evaluation."""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import write_model, Camera, Image, rotmat2qvec
R_ = Path(__import__("os").environ.get("LANE_SURVEY", "/home/paperspace/data/klapmuts/apr_2026_zed")); FX, FY, CX, CY, W, H = 527.985, 527.88, 638.975, 333.1835, 1280, 720
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["lo", "chunk"]); ap.add_argument("lane"); ap.add_argument("ref", nargs="?", default="transforms_ref_lo.json")
ap.add_argument("--proj", default="h3dgs"); ap.add_argument("--keyframes", action="store_true"); ap.add_argument("--kf-dist", type=float, default=0.20); ap.add_argument("--kf-deg", type=float, default=3.0)
ap.add_argument("--every", type=int, default=0, help="ablation (Paul 2026-09-27 'and then every fifth frame'): train on every N-th stream frame (stream index %% N == --offset; the every-10th held-out frames are unchanged)")
ap.add_argument("--resample-cm", type=float, default=0.0, help="2026-10-06 (Paul: 'is April velocity slower, why the frames'): April lane 2 was driven at 0.86 m/s vs December's 1.32 (5.9 vs 9.5 cm per frame at 15 Hz); resample the stream by arc length to this spacing (December: 9.5) BEFORE the every-10th held-out / --every rules, so both lanes get the same view geometry")
ap.add_argument("--offset", type=int, default=2, help="residue of the stream index kept by --every (2 keeps the training frames >= 2 frames from any held-out frame)")
a = ap.parse_args(); mode, LD = a.mode, Path(a.lane); stamps = {k: float(v) for k, v in json.load(open(LD / "stamps.json")).items()}; GL = np.diag([1.0, -1.0, -1.0, 1.0])
if mode == "lo":
    z = np.load(R_ / "experimental/laser_dump/lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]; slerp = Slerp(ts, Rotation.from_matrix(T[:, :3, :3]))
    C2L = np.linalg.inv(np.array(json.load(open(R_ / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64)); frames = []
    for name in sorted(stamps):
        t = float(np.clip(stamps[name], ts[0], ts[-1])); i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1); w = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
        M = np.eye(4); M[:3, :3] = slerp([t]).as_matrix()[0]; M[:3, 3] = (1 - w) * T[i - 1, :3, 3] + w * T[i, :3, 3]
        frames.append({"file_path": str(LD / "images" / name), "transform_matrix": ((M @ C2L) @ GL).tolist(), "timestamp_ns": int(stamps[name])})
    J = {"fl_x": FX, "fl_y": FY, "cx": CX, "cy": CY, "w": W, "h": H, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV", "ply_file_path": "init_lidar.ply",
         "pose_convention": "opengl_c2w (KISS-ICP LiDAR odometry slerp'd to the frame stamp x laser->camera extrinsic; metric)", "frames": frames}
    (LD / "transforms_lo.json").write_text(json.dumps(J, indent=1)); print(f"[lane-prep] transforms_lo.json: {len(frames)} frames", flush=True)
elif mode == "chunk":
    J = json.load(open(LD / a.ref)); P = LD / a.proj; CC = P / "camera_calibration"; CH = CC / "chunks/lane"; SP = CH / "sparse/0"
    for d in (SP, CC / "rectified/images", CC / "aligned/sparse/0", P / "output/trained_chunks"): d.mkdir(parents=True, exist_ok=True)
    allf = sorted(J["frames"], key=lambda f: f["file_path"])
    if a.resample_cm:   # nearest frame to every multiple of the spacing along the camera path (arc length)
        Cc = np.array([np.asarray(f["transform_matrix"], np.float64)[:3, 3] for f in allf]); arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cc, axis=0), axis=1))]
        pick = sorted(set(int(np.argmin(np.abs(arc - d))) for d in np.arange(0, arc[-1] + 1e-9, a.resample_cm / 100))); n0 = len(allf); allf = [allf[i] for i in pick]
        print(f"[lane-prep] resampled {n0} frames to {len(allf)} at {a.resample_cm:g} cm along {arc[-1]:.1f} m of path", flush=True)
    names = [Path(f["file_path"]).name for f in allf]; test = names[::10]; tset = set(test)
    keep = set(names)
    if a.keyframes:   # survey keyframe rule over the training candidates only (held-out frames never count as "last kept")
        keep = set(test); last = None
        for f in allf:
            n = Path(f["file_path"]).name
            if n in tset: continue
            M = np.asarray(f["transform_matrix"], np.float64)
            if last is None or np.linalg.norm(M[:3, 3] - last[:3, 3]) >= a.kf_dist or np.degrees(np.arccos(np.clip((np.trace(last[:3, :3].T @ M[:3, :3]) - 1) / 2, -1, 1))) >= a.kf_deg:
                keep.add(n); last = M
    if a.every:   # every N-th frame by list position (the held-out frames are positions % 10 == 0, so --offset 2 never collides with them)
        keep = set(test) | {n for p, n in enumerate(names) if p % a.every == a.offset % a.every}
    cams = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=np.array([FX, FY, CX, CY]))}; ims = {}; C = []
    for i, f in enumerate([f for f in allf if Path(f["file_path"]).name in keep], 1):
        name = Path(f["file_path"]).name; c2w = np.asarray(f["transform_matrix"], np.float64) @ GL; w2c = np.linalg.inv(c2w); C.append(c2w[:3, 3])
        ims[i] = Image(id=i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64))
        dst = CC / "rectified/images" / name
        if not dst.exists(): os.link(f["file_path"], dst)
    write_model(cams, ims, {}, str(SP), ".bin"); C = np.array(C); ctr = (C.max(0) + C.min(0)) / 2; ext = (C.max(0) - C.min(0)) + 12.0   # cell = lane bbox + 6 m each side
    np.savetxt(CH / "center.txt", ctr); np.savetxt(CH / "extent.txt", ext)
    (SP / "test.txt").write_text("\n".join(test) + "\n"); (CC / "aligned/sparse/0/test.txt").write_text("\n".join(test) + "\n"); write_model(cams, ims, {}, str(CC / "aligned/sparse/0"), ".bin")
    ply = LD / J.get("ply_file_path", "init_lidar.ply")
    if ply.exists(): shutil.copy2(ply, SP / "points3D.ply")
    json.dump({"survey_root": str(R_), "n_images": len(ims), "n_test": len(test), "n_train": len(ims) - len(test), "keyframes_only": a.keyframes, "resample_cm": a.resample_cm, "every": a.every, "camera": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H}, "sky_masks": None, "fg_masks": None,
               "world_rotation_to_zup": np.eye(4).tolist(), "pose_convention": "COLMAP w2c (OpenCV) in the LiDAR-odometry world (metric, z up)"}, open(P / "export_meta.json", "w"), indent=1)
    print(f"[lane-prep] {a.proj} chunk 'lane': {len(ims)} images ({len(ims) - len(test)} training{' = keyframes' if a.keyframes else ''}{f' = every {a.every}th frame' if a.every else ''}, {len(test)} held-out of {len(names)} frames), cell centre {ctr.round(2).tolist()} extent {ext.round(1).tolist()} m, LiDAR init {'yes' if ply.exists() else 'NO'} -> {P}", flush=True)
