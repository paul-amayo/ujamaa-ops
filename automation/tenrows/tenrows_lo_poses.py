"""Camera poses for the ten_rows keyframes from the KISS-ICP LiDAR odometry (2026-09-26): T_wc(t) = T_wl(t) @ L2C^-1
with T_wl the LiDAR pose (lo_poses.npz, 10 Hz, camera clock) slerp/lerp-interpolated at the keyframe stamp and L2C the
rig's laser->camera extrinsic (rig.json laser_to_camera_left: p_cam = L2C p_laser). Writes, for every canonical block,
block_NNN/transforms_lo.json (the block's frames, OpenGL c2w like the other transforms files) — the LO-placed odometry
that the per-block from-scratch SfM is then Sim(3)-aligned onto (klapmuts_apply_refine_named.py) — plus
experimental/lo_kf_poses.json {name: c2w_cv} for the LiDAR lifts.
  python (h3dgs env: scipy) tenrows_lo_poses.py"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R_ / "prod/monos/monolithics"; BL = R_ / "prod/tassili/blocks_ns/lio_row100"
z = np.load(R_ / "experimental/laser_dump/lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]
L2C = np.array(json.load(open(R_ / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64); C2L = np.linalg.inv(L2C)
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(MD / "kf_index.json"))}
rot = Rotation.from_matrix(T[:, :3, :3]); slerp = Slerp(ts, rot); GL = np.diag([1.0, -1.0, -1.0, 1.0])
def lo_pose(t):
    t = float(np.clip(t, ts[0], ts[-1])); i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1); w = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
    M = np.eye(4); M[:3, :3] = slerp([t]).as_matrix()[0]; M[:3, 3] = (1 - w) * T[i - 1, :3, 3] + w * T[i, :3, 3]; return M
allposes = {}; n_blocks = 0; gaps = []
for b in sorted(x for x in BL.glob("block_[0-9][0-9][0-9]") if x.name[6:].isdigit()):
    J = json.load(open(b / "transforms.json")); J2 = json.loads(json.dumps(J))
    for f in J2["frames"]:
        k = int(re.search(r"kf_(\d+)", f["file_path"]).group(1)); t = kf[k]; gaps.append(min(abs(ts - t)))
        c2w_cv = lo_pose(t) @ C2L; allposes[Path(f["file_path"]).name] = c2w_cv.tolist(); f["transform_matrix"] = (c2w_cv @ GL).tolist()
    J2["pose_convention"] = "opengl_c2w (KISS-ICP LiDAR odometry slerp'd to the keyframe stamp x laser->camera extrinsic; metric)"; J2["ply_file_path"] = "init_lidar.ply"
    (b / "transforms_lo.json").write_text(json.dumps(J2, indent=1)); n_blocks += 1
json.dump(allposes, open(R_ / "experimental/lo_kf_poses.json", "w"))
g = np.array(gaps); print(f"[lo-poses] {n_blocks} blocks, {len(allposes)} keyframes; nearest LO sample gap median {np.median(g):.0f} ms max {g.max():.0f} ms; L2C translation {np.round(L2C[:3, 3], 3).tolist()}; wrote block_NNN/transforms_lo.json + experimental/lo_kf_poses.json", flush=True)
