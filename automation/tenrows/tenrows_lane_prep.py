"""Single-lane H3DGS build for ten_rows (Paul, 2026-09-26: "prove in one row and then scale to the rest"). Two modes:
  lo    <lane dir>                : transforms_lo.json for the lane's full-stream frames (KISS-ICP laser pose slerp'd at
                                    each stamp x L2C^-1, OpenGL c2w, ZED-conf intrinsics) — the base the lane's own SfM
                                    is Sim(3)-placed on (klapmuts_apply_refine_orient.py)
  chunk <lane dir> <ref transforms> : the H3DGS project layout under <lane dir>/h3dgs: camera_calibration/chunks/lane/
                                    sparse/0 (COLMAP bin model in the LO world, one PINHOLE camera; LiDAR init as
                                    points3D.ply; test.txt every 10th frame), center/extent = lane bounding box,
                                    rectified/images (hardlinks), aligned/sparse/0 (cameras + test.txt) and a minimal
                                    export_meta.json so the compact evaluator runs."""
import json, os, shutil, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import write_model, Camera, Image, rotmat2qvec
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); FX, FY, CX, CY, W, H = 527.985, 527.88, 638.975, 333.1835, 1280, 720
mode, LD = sys.argv[1], Path(sys.argv[2]); stamps = {k: float(v) for k, v in json.load(open(LD / "stamps.json")).items()}; GL = np.diag([1.0, -1.0, -1.0, 1.0])
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
    J = json.load(open(LD / sys.argv[3])); P = LD / "h3dgs"; CC = P / "camera_calibration"; CH = CC / "chunks/lane"; SP = CH / "sparse/0"
    for d in (SP, CC / "rectified/images", CC / "aligned/sparse/0", P / "output/trained_chunks"): d.mkdir(parents=True, exist_ok=True)
    cams = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=np.array([FX, FY, CX, CY]))}; ims = {}; C = []
    for i, f in enumerate(sorted(J["frames"], key=lambda f: f["file_path"]), 1):
        name = Path(f["file_path"]).name; c2w = np.asarray(f["transform_matrix"], np.float64) @ GL; w2c = np.linalg.inv(c2w); C.append(c2w[:3, 3])
        ims[i] = Image(id=i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64))
        dst = CC / "rectified/images" / name
        if not dst.exists(): os.link(f["file_path"], dst)
    write_model(cams, ims, {}, str(SP), ".bin"); C = np.array(C); ctr = (C.max(0) + C.min(0)) / 2; ext = (C.max(0) - C.min(0)) + 12.0   # cell = lane bbox + 6 m each side
    np.savetxt(CH / "center.txt", ctr); np.savetxt(CH / "extent.txt", ext)
    names = [Path(f["file_path"]).name for f in sorted(J["frames"], key=lambda f: f["file_path"])]; test = names[::10]
    (SP / "test.txt").write_text("\n".join(test) + "\n"); (CC / "aligned/sparse/0/test.txt").write_text("\n".join(test) + "\n"); write_model(cams, ims, {}, str(CC / "aligned/sparse/0"), ".bin")
    ply = LD / J.get("ply_file_path", "init_lidar.ply")
    if ply.exists(): shutil.copy2(ply, SP / "points3D.ply")
    json.dump({"survey_root": str(R_), "n_images": len(ims), "n_test": len(test), "camera": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H}, "sky_masks": None, "fg_masks": None,
               "world_rotation_to_zup": np.eye(4).tolist(), "pose_convention": "COLMAP w2c (OpenCV) in the LiDAR-odometry world (metric, z up)"}, open(P / "export_meta.json", "w"), indent=1)
    print(f"[lane-prep] chunk 'lane': {len(ims)} images, {len(test)} held-out, cell centre {ctr.round(2).tolist()} extent {ext.round(1).tolist()} m, LiDAR init {'yes' if ply.exists() else 'NO'} -> {P}", flush=True)
