"""Keyframe cut on the LiDAR-odometry poses with a distance-OR-angle rule, process A (reader side: aru_nerf_interface
only, PIL first). Camera pose per left image = LO laser pose (lo_poses.npz, slerp at the image stamp) x L2C^-1 (rig.json).
Keep image 0; keep image i when it is > MIN_DIST from the last keyframe OR rotated > MIN_ANGLE from it (Paul, 09-26:
"oversample the turns"). Writes prod/scratch_sam3_lo/kf_%06d.png, monolithics/kf_index_lo.json (K, image_idx, ts_ms,
pos) and monolithics/lo_image_poses_laser.npz (ts_ms + laser pose for EVERY image, for process B's transform stream).
Nothing in the current keyframe set is touched (swap afterwards).
  env -u LD_LIBRARY_PATH -u LD_PRELOAD <nerf_new python3.10> tenrows_lo_kfcut_A.py [MIN_DIST=0.20] [MIN_ANGLE_DEG=3]"""
from PIL import Image  # noqa (PIL before the binding)
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
import aru_nerf_interface as a
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"; OUT = R / "prod/scratch_sam3_lo"; OUT.mkdir(parents=True, exist_ok=True)
MIN_DIST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20; MIN_ANGLE = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
z = np.load(R / "experimental/laser_dump/lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]
L2C = np.array(json.load(open(R / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64); C2L = np.linalg.inv(L2C)
rdr = a.ImageMonoReader(image_mono=str(MD / "image_left.monolithic")); n = rdr.num_images(); its = np.array([rdr.timestamp(i) for i in range(n)], float)
print(f"[kfcut-lo] {n} images {its[0]:.0f}..{its[-1]:.0f} ms; LO {len(ts)} poses {ts[0]:.0f}..{ts[-1]:.0f} ms; rule > {MIN_DIST} m OR > {MIN_ANGLE} deg", flush=True)
slerp = Slerp(ts, Rotation.from_matrix(T[:, :3, :3])); tq = np.clip(its, ts[0], ts[-1])
Rl = slerp(tq).as_matrix(); Pl = np.stack([np.interp(tq, ts, T[:, i, 3]) for i in range(3)], 1)
Twl = np.tile(np.eye(4), (n, 1, 1)); Twl[:, :3, :3] = Rl; Twl[:, :3, 3] = Pl
cam = Twl @ C2L; P = cam[:, :3, 3]; Rc = cam[:, :3, :3]
kept = [0]; last = 0
for i in range(1, n):
    d = np.linalg.norm(P[i] - P[last]); ang = np.degrees(np.arccos(np.clip((np.trace(Rc[last].T @ Rc[i]) - 1) / 2, -1, 1)))
    if d > MIN_DIST or ang > MIN_ANGLE: kept.append(i); last = i
outside = int(((its < ts[0]) | (its > ts[-1])).sum())
print(f"[kfcut-lo] {n} images -> {len(kept)} keyframes ({outside} images outside the LO span were clamped)", flush=True)
index = [{"K": K, "image_idx": int(i), "ts_ms": int(its[i]), "pos": P[i].round(4).tolist()} for K, i in enumerate(kept)]
(MD / "kf_index_lo.json").write_text(json.dumps(index)); np.savez(MD / "lo_image_poses_laser.npz", ts_ms=its.astype(np.int64), T=Twl, kept=np.array(kept))
for K, i in enumerate(kept):
    f = OUT / f"kf_{K:06d}.png"
    if f.exists() and f.stat().st_size > 0: continue
    img = np.asarray(rdr.read_index(i))[:, :, :3][:, :, ::-1]   # the aru_nerf_interface reader yields cv2 BGR; PIL saves RGB (the 09-05 and 09-26 swaps came from omitting this)
    Image.fromarray(np.ascontiguousarray(img)).save(f, compress_level=1)
    if K % 300 == 0: print(f"  kf {K}/{len(kept)}", flush=True)
print(f"[kfcut-lo] wrote {len(kept)} PNGs to {OUT} + kf_index_lo.json + lo_image_poses_laser.npz", flush=True)
