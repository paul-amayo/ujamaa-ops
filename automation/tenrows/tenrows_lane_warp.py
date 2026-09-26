"""Place a lane's own SfM onto the LiDAR-odometry track when a rigid Sim(3) is not enough (a 45 m lane solved with a fixed
camera comes out bent by ~1.5 m at the ends): global Sim(3) first, then a smooth correction field along the frame order
(Gaussian-weighted local LINEAR fit of the residual, sigma frames) so the cameras follow the LO at long range and keep
the SfM's local geometry, and a local yaw about the world up axis from a windowed rigid fit (roll/pitch left to the
SfM). Same construction as tenrows_ins_anchor.py --mode warp. Writes <lane dir>/transforms_ref_lo.json.
  python (h3dgs env) tenrows_lane_warp.py <lane dir> [<sfm transforms.json>] [--sigma 15]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("lane"); ap.add_argument("sfm", nargs="?", default=""); ap.add_argument("--sigma", type=float, default=15.0); ap.add_argument("--out", default="transforms_ref_lo.json")
a = ap.parse_args(); LD = Path(a.lane); SFM = Path(a.sfm) if a.sfm else LD / "colmap/transforms.json"
S = json.load(open(SFM)); Lo = json.load(open(LD / "transforms_lo.json"))
sfm = {Path(f["file_path"]).name: np.asarray(f["transform_matrix"], np.float64) for f in S["frames"]}; lo = {Path(f["file_path"]).name: np.asarray(f["transform_matrix"], np.float64) for f in Lo["frames"]}
names = sorted(n for n in lo if n in sfm); P = np.array([sfm[n][:3, 3] for n in names]); Q = np.array([lo[n][:3, 3] for n in names]); Rs = np.array([sfm[n][:3, :3] for n in names])
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    Rm = Vt.T @ Sg @ U.T; s = np.trace(np.diag(D) @ Sg) / (Xc ** 2).sum() * len(X); return s, Rm, my - s * Rm @ mx
s, Rm, t = umeyama(P, Q); C = s * (P @ Rm.T) + t; Rc = np.einsum("ij,njk->nik", Rm, Rs)          # SfM in the LO world (rigid)
r0 = np.linalg.norm(Q - C, axis=1)
def loess(vals, sigma):
    n = len(vals); idx = np.arange(n, dtype=np.float64); out = np.zeros_like(vals)
    for i in range(n):
        d = idx - i; w = np.exp(-0.5 * (d / sigma) ** 2); sw = w.sum(); m1 = (w * d).sum() / sw; v = (w * d * d).sum() / sw - m1 * m1
        if v < 1e-9: out[i] = (w[:, None] * vals).sum(0) / sw; continue
        beta = ((w * (d - m1))[:, None] * vals).sum(0) / (sw * v); out[i] = (w[:, None] * vals).sum(0) / sw - beta * m1
    return out
corr = loess(Q - C, a.sigma); Cn = C + corr
up = np.array([0.0, 0.0, 1.0]); u1 = np.array([1.0, 0, 0]); u2 = np.cross(up, u1)
c2 = np.stack([C @ u1, C @ u2], 1); q2 = np.stack([Q @ u1, Q @ u2], 1); th = np.zeros(len(C)); idx = np.arange(len(C), dtype=np.float64)
for i in range(len(C)):
    w = np.exp(-0.5 * ((idx - i) / a.sigma) ** 2); w /= w.sum(); cm = (w[:, None] * c2).sum(0); qm = (w[:, None] * q2).sum(0); X = c2 - cm; Y = q2 - qm; Hm = (w[:, None] * X).T @ Y
    th[i] = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
def rz(a_): c, s_ = np.cos(a_), np.sin(a_); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
frames = []; byname = {Path(f["file_path"]).name: f for f in Lo["frames"]}
for i, n in enumerate(names):
    M = np.eye(4); M[:3, :3] = rz(th[i]) @ Rc[i]; M[:3, 3] = Cn[i]; frames.append(dict(byname[n], transform_matrix=M.tolist()))
r1 = np.linalg.norm(Q - Cn, axis=1); seg = np.array_split(np.arange(len(r1)), 8)
out = dict(Lo); out["frames"] = frames; out["pose_convention"] = f"opengl_c2w (lane SfM warped onto the LiDAR odometry: Sim(3) + LOESS position field sigma {a.sigma:g} frames + windowed yaw)"
(LD / a.out).write_text(json.dumps(out, indent=1))
print(f"[lane-warp] {len(names)} frames; Sim(3) scale {s:.3f}: residual to LO p50 {np.median(r0):.3f} -> {np.median(r1):.3f} m (p90 {np.percentile(r0,90):.3f} -> {np.percentile(r1,90):.3f}); by segment after {[round(float(np.median(r1[i])), 3) for i in seg]}; yaw correction median {np.degrees(np.median(np.abs(th))):.2f} max {np.degrees(np.abs(th).max()):.2f} deg -> {LD / a.out}", flush=True)
