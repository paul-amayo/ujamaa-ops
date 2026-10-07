"""EXPORT POSES ONLY (2026-10-06, for the top-down: the lane-LO -> export Sim(3) needs the export pose of every lane frame; writes <lane>/transforms_lo_export.json and nothing else). From apr_lane_base.py: April Klapmuts lane 2, base poses + LiDAR init for the lane recipe (Paul 2026-10-06: "run it now"). December's lane recipe
(tenrows_lane_run.sh) places the lane's own SfM onto a metric base (KISS-ICP LO from ten_rows' laser_dump) and seeds H3DGS with
the raw laser scans. April has no laser_dump; its metric camera poses are the April H3DGS export's keyframe poses (LIO-derived,
globally BA'd, z-up, the frame the April chunks were trained in), so:
  base  every full-stream frame (stamps.json) gets the export pose slerped between the bracketing KEYFRAMES (keyframe stamps
        read from image_left_kf20cm.monolithic, record index = kf number) -> <lane>/transforms_lo.json (OpenGL c2w, April
        rig.json intrinsics, the December 'lo' mode's layout)
  lidar every 3rd lane frame's nearest laser scan (laser.monolithic, <= 60 ms) placed with world_from_cam @ laser_to_camera_left,
        cropped to the lane box + pad (6, 6, 4 m), 5 cm voxels, coloured from the frame -> <lane>/init_lidar.ply
  nerf_new python3.10 + aru_py_logger: apr_lane_base.py <lane dir>"""
import json, re, sys
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
S = Path("/home/paperspace/data/klapmuts/apr_2026_zed"); M = S / "prod/monos"; EXP = S / "experimental/h3dgs"; LD = Path(sys.argv[1])
rig = json.load(open(M / "rig.json")); L2C = np.array(json.loads(rig["laser_to_camera_left"]) if isinstance(rig["laser_to_camera_left"], str) else rig["laser_to_camera_left"], np.float64)
intr = rig["intrinsics"]; intr = json.loads(intr.replace("'", '"')) if isinstance(intr, str) else intr
FX, FY, CX, CY, W, H = intr["fx"], intr["fy"], intr["cx"], intr["cy"], int(intr["img_w"]), int(intr["img_h"])
stamps = {k: float(v) for k, v in json.load(open(LD / "stamps.json")).items()}; win = json.load(open(LD / "window.json"))
# keyframe stamps (record index = kf number) over the window +- 5 keyframes
k0, k1 = max(0, win["kf_first"] - 5), win["kf_last"] + 5; lg = aru_py_logger.MonoImageLogger(str(M / "image_left_kf20cm.monolithic"), False); kts = {}; i = 0
while not lg.end_of_file():
    img, ts = lg.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    if k0 <= i <= k1: kts[i] = float(ts)
    if i > k1: break
    i += 1
ims = {im.name: im for im in read_images_binary(str(EXP / "camera_calibration/aligned/sparse/0/images.bin")).values()}
K = sorted(k for k in kts if f"kf_{k:06d}.png" in ims); T = []
for k in K:
    im = ims[f"kf_{k:06d}.png"]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; T.append(np.linalg.inv(w2c))   # c2w, OpenCV, z-up H3DGS frame
T = np.array(T); kt = np.array([kts[k] for k in K]); slerp = Slerp(kt, Rotation.from_matrix(T[:, :3, :3])); GL = np.diag([1.0, -1.0, -1.0, 1.0]); frames = []; C2W = {}
for name in sorted(stamps, key=lambda n: int(re.sub(r"\D", "", n))):
    t = float(np.clip(stamps[name], kt[0], kt[-1])); j = int(np.clip(np.searchsorted(kt, t), 1, len(kt) - 1)); w = (t - kt[j - 1]) / max(kt[j] - kt[j - 1], 1e-9)
    c = np.eye(4); c[:3, :3] = slerp([t]).as_matrix()[0]; c[:3, 3] = (1 - w) * T[j - 1, :3, 3] + w * T[j, :3, 3]; C2W[name] = c
    frames.append({"file_path": str(LD / "images" / name), "transform_matrix": (c @ GL).tolist(), "timestamp_ns": int(stamps[name])})
J = {"fl_x": FX, "fl_y": FY, "cx": CX, "cy": CY, "w": W, "h": H, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV", "ply_file_path": "init_lidar.ply",
     "pose_convention": "opengl_c2w (April H3DGS export keyframe poses, z-up metric, slerped to the frame stamp)", "frames": frames}
(LD / "transforms_lo_export.json").write_text(json.dumps(J, indent=1))
step = np.median(np.linalg.norm(np.diff(np.array([C2W[n][:3, 3] for n in sorted(C2W, key=lambda n: int(re.sub(r'\D', '', n)))]), axis=0), axis=1))
print(f"[apr-export] transforms_lo_export.json: {len(frames)} frames from {len(K)} keyframes ({K[0]}..{K[-1]}); median step {step * 100:.1f} cm; intrinsics {FX:.2f} {FY:.2f} {CX:.2f} {CY:.2f}", flush=True)
