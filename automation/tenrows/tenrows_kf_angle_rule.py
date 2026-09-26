"""Keyframe rule study for ten_rows (Paul, 2026-09-26: "our keyframe approach takes distance but we should also have
like minimal angles, so that we oversample the turns"). Current rule (tenrows_kfcut_A.py / dump_keyframe_image_monolithic):
keep image i when the straight-line distance from the last kept keyframe > 0.20 m. Candidate: OR the rotation since the
last keyframe > theta. Poses for every left image (5,903 stamps from the mcap) come from the KISS-ICP LiDAR odometry
(slerp) — no ZED involved. Reports keyframe counts, keyframes inside turns (LO yaw rate > 8 deg/s) and the largest
rotation between consecutive keyframes, per rule.
  python (h3dgs env: scipy) tenrows_kf_angle_rule.py"""
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows")
z = np.load(R_ / "experimental/laser_dump/lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); T = z["T"]
img = np.load(R_ / "prod/tassili/ten_rows_ins_stamps.npz")["left"][:, 0] / 1e6      # header stamps, ms (camera clock)
img = np.sort(img); img = img[(img >= ts[0]) & (img <= ts[-1])]
rot = Rotation.from_matrix(T[:, :3, :3]); slerp = Slerp(ts, rot); Rq = slerp(img); Pq = np.stack([np.interp(img, ts, T[:, 3 - 3 + i, 3]) for i in range(3)], 1)
yaw = np.unwrap(Rq.as_euler("zyx")[:, 0]); dt = np.gradient(img) / 1e3; yaw_rate = np.degrees(np.abs(np.gradient(yaw) / dt)); k = 15; yaw_rate = np.convolve(yaw_rate, np.ones(k) / k, mode="same")
turn = yaw_rate > 8.0
print(f"[kf-rule] {len(img)} images over {(img[-1] - img[0]) / 1e3:.0f} s; frames in turns (LO yaw rate > 8 deg/s, 1 s smoothing): {turn.sum()} ({turn.mean():.0%}), turn segments ≈ {int(np.sum(np.diff(turn.astype(int)) == 1))}", flush=True)
def cut(min_dist, theta_deg):
    kept = [0]; last_p, last_r = Pq[0], Rq[0]
    for i in range(1, len(img)):
        d = np.linalg.norm(Pq[i] - last_p); a = np.degrees((last_r.inv() * Rq[i]).magnitude())
        if d > min_dist or (theta_deg is not None and a > theta_deg): kept.append(i); last_p, last_r = Pq[i], Rq[i]
    kept = np.array(kept); rel = np.array([np.degrees((Rq[a].inv() * Rq[b]).magnitude()) for a, b in zip(kept[:-1], kept[1:])])
    return kept, rel
print("[kf-rule] rule                     keyframes   in turns   rotation between consecutive keyframes: median / p90 / max (deg)")
for theta in (None, 10.0, 8.0, 5.0, 3.0, 2.0):
    kept, rel = cut(0.20, theta); label = "20 cm only (current)" if theta is None else f"20 cm OR {theta:g} deg"
    print(f"[kf-rule] {label:26s} {len(kept):6d}   {int(turn[kept].sum()):6d}     {np.median(rel):.2f} / {np.percentile(rel, 90):.2f} / {rel.max():.2f}", flush=True)
