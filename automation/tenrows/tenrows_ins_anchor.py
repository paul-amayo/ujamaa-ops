"""INS-anchored block poses for dec_2025_ten_rows (2026-09-25). Each block keeps its image-refined poses
(transforms_ref_zedconf.json: per-block fixed-camera GLOMAP, Sim(3)-aligned into the LIO world) at short range, but is
pulled onto camera centres derived from the INS (Fixposition /ins_0/odom; clock +dt s ahead of the cameras; INS->camera
lever arm fitted globally) at long range, so the lanes sit where the INS says and the seams between consecutive blocks
close. Two modes:
  rigid : per block one yaw (about the world up axis) + translation + scale (the first attempt; leaves/widens seams)
  warp  : per block a smooth correction field along the keyframe order (Gaussian, --sigma keyframes, truncated at the
          block boundary): position = image-refined + smoothed (INS-derived - image-refined); orientation rotated about
          up by the angle between the smoothed tangents before/after. The LiDAR init moves with its nearest keyframe.
Writes block_NNN/transforms_ref_ins.json and block_NNN_ins/{init_lidar.ply,transforms.json} (suffixed dir = experiment
per the prod doctrine). Nothing existing is modified.
  python (h3dgs env: plyfile, scipy) tenrows_ins_anchor.py [--mode warp] [--sigma 10] [--dt 2.385] [--dry]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R_ / "prod/monos/monolithics"; BL = R_ / "prod/tassili/blocks_ns/lio_row100"
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["rigid", "warp"], default="warp"); ap.add_argument("--sigma", type=float, default=10.0)
ap.add_argument("--dt", type=float, default=2.385); ap.add_argument("--src", default="transforms_ref_zedconf.json"); ap.add_argument("--out", default="transforms_ref_ins.json")
ap.add_argument("--dry", action="store_true"); ap.add_argument("--min_frames", type=int, default=20)
a = ap.parse_args()

def traj(path):
    T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in _read_increments(Path(path)):
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m
        T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    ts = np.asarray(ts, np.float64); ts = ts / 1e6 if ts.max() > 1e15 else ts; return ts, np.array(P), np.array(Rs)   # ms

def interp(ts_src, P_src, R_src, ts_q):
    i = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1); t0, t1 = ts_src[i - 1], ts_src[i]
    w = np.clip((ts_q - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]; near = np.where((ts_q - t0) < (t1 - ts_q), i - 1, i)
    return (1 - w) * P_src[i - 1] + w * P_src[i], R_src[near]

def umeyama(X, Y, with_scale=True):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(X.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[-1, -1] = -1
    Rm = Vt.T @ S @ U.T; s = (np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)) if with_scale else 1.0
    return s, Rm, my - s * Rm @ mx

def global_fit(C, P, Rins, iters=8):
    l = np.zeros(3)
    for _ in range(iters):
        Q = P + np.einsum("nij,j->ni", Rins, l); s, Rm, t = umeyama(Q, C)
        A = s * np.einsum("ij,njk->nik", Rm, Rins).reshape(-1, 3); b = (C - s * (P @ Rm.T) - t).reshape(-1); l = np.linalg.lstsq(A, b, rcond=None)[0]
    return s, Rm, t, l

def rodrigues(axis, th):
    k = axis / np.linalg.norm(axis); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def smooth_field(vals, ok, sigma):
    """Gaussian-weighted LOCAL LINEAR fit along the frame order (LOESS degree 1) using only the covered frames (ok):
    unlike a plain weighted mean it does not lag a linear trend at the block ends, so the two sides of a seam both land
    on the INS. Every frame gets a value."""
    n = len(vals); idx = np.arange(n); out = np.zeros_like(vals, dtype=np.float64); src = np.where(ok)[0]
    for i in range(n):
        d = (idx[src] - i).astype(np.float64); w = np.exp(-0.5 * (d / sigma) ** 2); sw = w.sum()
        m1 = (w * d).sum() / sw; v = (w * d * d).sum() / sw - m1 * m1
        if v < 1e-9: out[i] = (w[:, None] * vals[src]).sum(0) / sw; continue
        beta = ((w * (d - m1))[:, None] * vals[src]).sum(0) / (sw * v); alpha = (w[:, None] * vals[src]).sum(0) / sw - beta * m1
        out[i] = alpha
    return out

def local_yaw(C, T, ok, sigma, up):
    """Per-frame yaw correction = rotation about `up` of the Gaussian-weighted rigid (Procrustes) fit of the image-refined
    centres onto the INS-derived ones in the ground plane, window sigma frames, covered frames only."""
    u1 = np.cross(up, [1.0, 0, 0]); u1 = u1 if np.linalg.norm(u1) > 0.1 else np.cross(up, [0, 1.0, 0]); u1 /= np.linalg.norm(u1); u2 = np.cross(up, u1)
    c2 = np.stack([C @ u1, C @ u2], 1); t2 = np.stack([T @ u1, T @ u2], 1); src = np.where(ok)[0]; n = len(C); th = np.zeros(n)
    for i in range(n):
        w = np.exp(-0.5 * ((src - i) / sigma) ** 2); w /= w.sum(); cm = (w[:, None] * c2[src]).sum(0); tm = (w[:, None] * t2[src]).sum(0)
        X = c2[src] - cm; Y = t2[src] - tm; H = (w[:, None] * X).T @ Y                      # 2x2 weighted cross-covariance
        th[i] = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])                             # rotation angle of the 2D Procrustes solution
    return th

# INS at the keyframes
ts_ins, P_ins, R_ins = traj(MD / "ins_transform.monolithic"); up_ins = np.zeros(3); up_ins[int(np.argmin(np.ptp(P_ins, 0)))] = 1.0
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(MD / "kf_index.json"))}
blocks = sorted(b for b in BL.glob("block_[0-9][0-9][0-9]") if b.name[6:].isdigit() and (b / a.src).exists())
print(f"[anchor] mode {a.mode} (sigma {a.sigma:g} kf); {len(blocks)} blocks with {a.src}; INS {len(ts_ins)} poses; dt {a.dt:+.3f} s (INS clock ahead of the cameras)", flush=True)
data = []; allC, allP, allR = [], [], []
for b in blocks:
    J = json.load(open(b / a.src)); Ks = [int(re.search(r"kf_(\d+)", f["file_path"]).group(1)) for f in J["frames"]]
    order = np.argsort(Ks); J["frames"] = [J["frames"][i] for i in order]; Ks = [Ks[i] for i in order]          # keyframe order = travel order
    M = np.array([np.asarray(f["transform_matrix"], np.float64) for f in J["frames"]]); C = M[:, :3, 3].copy()
    tq = np.array([kf[k] for k in Ks]) + a.dt * 1e3; ok = (tq >= ts_ins[0]) & (tq <= ts_ins[-1])
    Pq, Rq = interp(ts_ins, P_ins, R_ins, tq); data.append([b, J, Ks, M, C, Pq, Rq, ok]); allC.append(C[ok]); allP.append(Pq[ok]); allR.append(Rq[ok])
C_all, P_all, R_all = np.concatenate(allC), np.concatenate(allP), np.concatenate(allR)
s_g, R_g, t_g, lever = global_fit(C_all, P_all, R_all)
up = R_g @ up_ins; up /= np.linalg.norm(up)
res_before = []; res_after = []; rows = []; written = 0; seam_pairs = []
for rec in data:
    b, J, Ks, M, C, Pq, Rq, ok = rec
    T = s_g * ((Pq + np.einsum("nij,j->ni", Rq, lever)) @ R_g.T) + t_g            # INS-derived centres in the LIO world
    if ok.sum() < a.min_frames: print(f"[anchor] {b.name}: only {int(ok.sum())} keyframes with INS coverage -> left as is"); rec.append(None); continue
    if a.mode == "rigid":
        u1 = np.cross(up, [1.0, 0, 0]); u1 /= np.linalg.norm(u1); u2 = np.cross(up, u1)
        c2 = np.stack([C[ok] @ u1, C[ok] @ u2], 1); t2 = np.stack([T[ok] @ u1, T[ok] @ u2], 1); s_b, R2, _ = umeyama(c2, t2)
        th = np.full(len(C), np.arctan2(R2[1, 0], R2[0, 0])); R_b = rodrigues(up, th[0]); t_b = T[ok].mean(0) - s_b * R_b @ C[ok].mean(0)
        Cn = s_b * (C @ R_b.T) + t_b; Rcorr = [R_b] * len(C); note = f"rigid yaw {np.degrees(th[0]):+.2f} deg scale {s_b:.4f}"
    else:
        r = np.where(ok[:, None], T - C, 0.0); rs = smooth_field(r, ok, a.sigma); Cn = C + rs
        th = local_yaw(C, T, ok, a.sigma, up); Rcorr = [rodrigues(up, x) for x in th]; note = f"warp sigma {a.sigma:g}"
    r0 = np.linalg.norm((C - T)[ok], axis=1); r1 = np.linalg.norm((Cn - T)[ok], axis=1); res_before.append(r0); res_after.append(r1)
    rows.append((b.name, len(Ks), int(ok.sum()), np.median(r0), np.median(r1), np.percentile(r1, 90), np.degrees(np.abs(th)).max(), np.linalg.norm(Cn - C, axis=1).max(), np.degrees(np.median(th))))
    rec.append((Cn, Rcorr, th, note))
print(f"[anchor] global Sim(3)+lever arm over {len(C_all)} keyframes: scale {s_g:.4f} LIO units per INS metre, lever arm (INS body) {np.round(lever, 3).tolist()} (|l| {np.linalg.norm(lever):.3f} m); up axis {np.round(up, 3).tolist()}")
print("[anchor] block   frames  n_ins   residual to INS median before -> after (p90 after)   yaw corr median / max|.| deg   max move (m)")
for r in rows: print(f"[anchor] {r[0]}  {r[1]:5d}  {r[2]:5d}   {r[3]:.3f} -> {r[4]:.3f} ({r[5]:.3f})   {r[8]:+6.2f} / {r[6]:5.2f}   {r[7]:6.3f}")
rb, ra = np.concatenate(res_before), np.concatenate(res_after)
print(f"[anchor] all covered keyframes: residual to INS median {np.median(rb):.3f} -> {np.median(ra):.3f}, p90 {np.percentile(rb, 90):.3f} -> {np.percentile(ra, 90):.3f}, max {rb.max():.3f} -> {ra.max():.3f} (LIO units)")
# seams: consecutive keyframes across block boundaries, before vs after
ends = {}
for rec in data:
    b, J, Ks, M, C, *_ = rec; fit = rec[-1]
    if fit is None: continue
    ends[Ks[0]] = ("start", C[0], fit[0][0]); ends[Ks[-1]] = ("end", C[-1], fit[0][-1])
seams = [(k, np.linalg.norm(ends[k + 1][1] - ends[k][1]), np.linalg.norm(ends[k + 1][2] - ends[k][2])) for k in ends if ends[k][0] == "end" and (k + 1) in ends and ends[k + 1][0] == "start"]
if seams: print(f"[anchor] {len(seams)} block seams (consecutive keyframes): gap median {np.median([s[1] for s in seams]):.2f} -> {np.median([s[2] for s in seams]):.2f} m, max {max(s[1] for s in seams):.2f} -> {max(s[2] for s in seams):.2f} m; after: {np.round(sorted(s[2] for s in seams), 2).tolist()}")
if a.dry: sys.exit(0)
for rec in data:
    b, J, Ks, M, C, Pq, Rq, ok, fit = rec
    if fit is None: continue
    Cn, Rcorr, th, note = fit
    J2 = json.loads(json.dumps(J)); J2["pose_convention"] = J.get("pose_convention", "") + f" | INS-anchored ({note}; dt {a.dt:+.3f} s, lever {np.round(lever, 3).tolist()}, global scale {s_g:.4f})"
    J2["ins_anchor"] = {"src": a.src, "mode": a.mode, "sigma_kf": a.sigma, "dt_s": a.dt, "lever_ins_body": lever.tolist(), "global_scale": float(s_g), "n_ins": int(ok.sum()),
                        "residual_median_before": float(np.median(np.linalg.norm((C - (s_g * ((Pq + np.einsum('nij,j->ni', Rq, lever)) @ R_g.T) + t_g))[ok], axis=1))),
                        "max_yaw_corr_deg": float(np.degrees(np.abs(th)).max())}
    for i, f in enumerate(J2["frames"]):
        M2 = np.eye(4); M2[:3, :3] = Rcorr[i] @ M[i, :3, :3]; M2[:3, 3] = Cn[i]; f["transform_matrix"] = M2.tolist()
    (b / a.out).write_text(json.dumps(J2, indent=1))
    vd = b.parent / (b.name + "_ins"); vd.mkdir(exist_ok=True); (vd / "transforms.json").write_text(json.dumps(J2, indent=1))
    ply = b / "init_lidar.ply"
    if ply.exists():   # every LiDAR point moves with its nearest keyframe: same translation, same yaw about that camera
        v = PlyData.read(str(ply))["vertex"]; xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        nn = cKDTree(C).query(xyz, k=1)[1]; xyz2 = np.empty_like(xyz)
        for j in np.unique(nn):
            m = nn == j; xyz2[m] = Cn[j] + (xyz[m] - C[j]) @ Rcorr[j].T
        arr = v.data.copy(); arr["x"], arr["y"], arr["z"] = xyz2[:, 0].astype(arr["x"].dtype), xyz2[:, 1].astype(arr["y"].dtype), xyz2[:, 2].astype(arr["z"].dtype)
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(vd / "init_lidar.ply"))
    written += 1
print(f"[anchor] wrote {written} x {a.out} + block_NNN_ins/{{init_lidar.ply,transforms.json}}", flush=True)
