"""Place a lane's own SfM onto the LiDAR-odometry track when a rigid Sim(3) is not enough (a 45 m lane solved with a fixed
camera comes out bent by ~1.5 m at the ends): global Sim(3) first, then a smooth correction field along the frame order
(Gaussian-weighted local LINEAR fit of the residual, sigma frames) so the cameras follow the LO at long range and keep
the SfM's local geometry, and a local yaw about the world up axis from a windowed rigid fit (roll/pitch left to the
SfM). Same construction as tenrows_ins_anchor.py --mode warp. Writes <lane dir>/transforms_ref_lo.json.
  python (h3dgs env) tenrows_lane_warp.py <lane dir> [<sfm transforms.json>] [--sigma 15] [--split-jumps 0.5]
--split-jumps M (2026-10-06, April lane 2): GLOMAP's global positioning on a straight forward-moving lane can break the chain
into pieces with different scales (April: 7.5 m and 5.6 m jumps between consecutive frames that share ~2,400 inliers; piece
scales 2.23 / 1.87 / 2.15). Cut the SfM wherever a consecutive step disagrees with the LO step by more than M metres (after
the global Sim(3) scale) and warp every piece on its own (Sim(3) + LOESS + windowed yaw inside the piece); pieces shorter
than --min-piece frames are dropped. 0 (default) = the December behaviour, one piece.
--roll-from-lo D (2026-10-06, April lane 2 full): a position-only Sim(3) cannot see a roll about the track of a short,
near-straight piece (April's 63-frame / 3.7 m middle piece came out rolled 50.4 deg against the LO; the heading check is
blind to it because the optical axis stays put). Per piece, take the twist about the piece's track axis of the mean rotation
from the placed cameras to the LO cameras; if it exceeds D degrees, apply it (centroid kept). The LO cameras carry the
laser->camera extrinsic error (2-4 deg in December), so D sits well above it; 0 (default) = off."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ap = argparse.ArgumentParser(); ap.add_argument("lane"); ap.add_argument("sfm", nargs="?", default=""); ap.add_argument("--sigma", type=float, default=15.0); ap.add_argument("--out", default="transforms_ref_lo.json")
ap.add_argument("--split-jumps", type=float, default=0.0); ap.add_argument("--roll-from-lo", type=float, default=0.0); ap.add_argument("--min-piece", type=int, default=30)
a = ap.parse_args(); LD = Path(a.lane); SFM = Path(a.sfm) if a.sfm else LD / "colmap/transforms.json"
S = json.load(open(SFM)); Lo = json.load(open(LD / "transforms_lo.json"))
sfm = {Path(f["file_path"]).name: np.asarray(f["transform_matrix"], np.float64) for f in S["frames"]}; lo = {Path(f["file_path"]).name: np.asarray(f["transform_matrix"], np.float64) for f in Lo["frames"]}
names = sorted(n for n in lo if n in sfm); P = np.array([sfm[n][:3, 3] for n in names]); Q = np.array([lo[n][:3, 3] for n in names]); Rs = np.array([sfm[n][:3, :3] for n in names])
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    Rm = Vt.T @ Sg @ U.T; s = np.trace(np.diag(D) @ Sg) / (Xc ** 2).sum() * len(X); return s, Rm, my - s * Rm @ mx
def loess(vals, sigma):
    n = len(vals); idx = np.arange(n, dtype=np.float64); out = np.zeros_like(vals)
    for i in range(n):
        d = idx - i; w = np.exp(-0.5 * (d / sigma) ** 2); sw = w.sum(); m1 = (w * d).sum() / sw; v = (w * d * d).sum() / sw - m1 * m1
        if v < 1e-9: out[i] = (w[:, None] * vals).sum(0) / sw; continue
        beta = ((w * (d - m1))[:, None] * vals).sum(0) / (sw * v); out[i] = (w[:, None] * vals).sum(0) / sw - beta * m1
    return out
def rz(a_): c, s_ = np.cos(a_), np.sin(a_); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
ROLLFIX = []
def align(P, Q, Rs, RL):   # one piece: Sim(3), LOESS position field, windowed yaw (the December construction, unchanged)
    s, Rm, t = umeyama(P, Q)
    if a.roll_from_lo > 0:   # the roll about the track axis that positions cannot see, from the LO cameras (only when it is gross)
        Rc0 = np.einsum("ij,njk->nik", Rm, Rs); U_, _, Vt_ = np.linalg.svd(np.einsum("nij,nkj->ik", RL, Rc0)); Rbar = U_ @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U_ @ Vt_))]) @ Vt_
        u = np.linalg.svd(Q - Q.mean(0))[2][0]; q = Rotation.from_matrix(Rbar).as_quat(); tw = 2 * np.arctan2(q[:3] @ u, q[3]); tw = (tw + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(tw)) > a.roll_from_lo:
            Rm = Rotation.from_rotvec(u * tw).as_matrix() @ Rm; t = Q.mean(0) - s * Rm @ P.mean(0); ROLLFIX.append(round(float(np.degrees(tw)), 1))
    C = s * (P @ Rm.T) + t; Rc = np.einsum("ij,njk->nik", Rm, Rs)          # SfM in the LO world (rigid)
    corr = loess(Q - C, a.sigma); Cn = C + corr
    up = np.array([0.0, 0.0, 1.0]); u1 = np.array([1.0, 0, 0]); u2 = np.cross(up, u1)
    c2 = np.stack([C @ u1, C @ u2], 1); q2 = np.stack([Q @ u1, Q @ u2], 1); th = np.zeros(len(C)); idx = np.arange(len(C), dtype=np.float64)
    for i in range(len(C)):
        w = np.exp(-0.5 * ((idx - i) / a.sigma) ** 2); w /= w.sum(); cm = (w[:, None] * c2).sum(0); qm = (w[:, None] * q2).sum(0); X = c2 - cm; Y = q2 - qm; Hm = (w[:, None] * X).T @ Y
        th[i] = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
    return s, C, Cn, Rc, th
RL = np.array([lo[n][:3, :3] for n in names]); s_g, Rm_g, t_g = umeyama(P, Q); r0 = np.linalg.norm(Q - (s_g * (P @ Rm_g.T) + t_g), axis=1); cuts = []
if a.split_jumps > 0:
    e = np.abs(s_g * np.linalg.norm(np.diff(P, axis=0), axis=1) - np.linalg.norm(np.diff(Q, axis=0), axis=1)); cuts = [int(i) + 1 for i in np.where(e > a.split_jumps)[0]]
pieces = [(b0, b1) for b0, b1 in zip([0] + cuts, cuts + [len(names)])]; keep = []; Cn = np.zeros_like(Q); C = np.zeros_like(Q); Rc = np.zeros_like(Rs); th = np.zeros(len(names)); scales = []
for b0, b1 in pieces:
    if b1 - b0 < a.min_piece: continue
    s, C[b0:b1], Cn[b0:b1], Rc[b0:b1], th[b0:b1] = align(P[b0:b1], Q[b0:b1], Rs[b0:b1], RL[b0:b1]); keep += list(range(b0, b1)); scales.append(round(float(s), 3))
s = s_g
frames = []; byname = {Path(f["file_path"]).name: f for f in Lo["frames"]}
for i in keep:
    n = names[i]; M = np.eye(4); M[:3, :3] = rz(th[i]) @ Rc[i]; M[:3, 3] = Cn[i]; frames.append(dict(byname[n], transform_matrix=M.tolist()))
r1 = np.linalg.norm(Q[keep] - Cn[keep], axis=1); seg = np.array_split(np.arange(len(r1)), 8); th = th[keep]
out = dict(Lo); out["frames"] = frames; out["pose_convention"] = f"opengl_c2w (lane SfM warped onto the LiDAR odometry: Sim(3) + LOESS position field sigma {a.sigma:g} frames + windowed yaw{f'; split at SfM jumps > {a.split_jumps:g} m into {len(scales)} pieces' if cuts else ''})"
(LD / a.out).write_text(json.dumps(out, indent=1))
print(f"[lane-warp] {len(keep)}/{len(names)} frames" + (f"; cuts before {[names[c] for c in cuts]} -> pieces {[(names[b0], b1 - b0) for b0, b1 in pieces]}, piece scales {scales}" if cuts else "") + (f"; roll fixed from the LO on {len(ROLLFIX)} piece(s): {ROLLFIX} deg" if ROLLFIX else "") + f"; Sim(3) scale {s:.3f}: residual to LO p50 {np.median(r0):.3f} -> {np.median(r1):.3f} m (p90 {np.percentile(r0,90):.3f} -> {np.percentile(r1,90):.3f}); by segment after {[round(float(np.median(r1[i])), 3) for i in seg]}; yaw correction median {np.degrees(np.median(np.abs(th))):.2f} max {np.degrees(np.abs(th).max()):.2f} deg -> {LD / a.out}", flush=True)
