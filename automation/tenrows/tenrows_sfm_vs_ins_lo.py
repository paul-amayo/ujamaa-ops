"""Trajectory comparison at the survey's keyframe stamps (Paul, 2026-09-26: "when sfm is done, compare it with the ins
and LO, let's see what the trajectory looks like"): the H3DGS project's COLMAP camera centres after the global BA
(aligned model), its prior (per-block SfM placed on the LiDAR odometry), the LiDAR odometry itself (KISS-ICP, camera
centre = T_wl x L2C^-1 at the keyframe stamp) and the INS (RTK-aided, clock +2.385 s ahead of the cameras).
Every source is Sim(3)-aligned onto the INS (lever arm fitted for the camera-centred sources), residual stats, drift vs
distance and adjacent-lane spacings; plus COLMAP-vs-LO directly (how far the image BA moved the cameras off the LiDAR
trajectory). Figure -> prod/tassili/ten_rows_sfm_vs_ins_lo.png.
  python (h3dgs env) tenrows_sfm_vs_ins_lo.py [<proj dir>]"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"; PROJ = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "experimental/h3dgs_lo3"
DT_S = 2.385
def traj(path):
    T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in _read_increments(Path(path)):
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m; T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    ts = np.asarray(ts, np.float64); ts = ts / 1e6 if ts.max() > 1e15 else ts; return ts, np.array(P), np.array(Rs)
def interp(ts_src, P_src, R_src, ts_q):
    i = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1); t0, t1 = ts_src[i - 1], ts_src[i]
    w = np.clip((ts_q - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]; near = np.where((ts_q - t0) < (t1 - ts_q), i - 1, i); return (1 - w) * P_src[i - 1] + w * P_src[i], R_src[near]
def umeyama(X, Y, with_scale=True):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = Vt.T @ S @ U.T; s = (np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)) if with_scale else 1.0; return s, Rm, my - s * Rm @ mx
def fit_lever(C, P, Rins, iters=8):
    l = np.zeros(3)
    for _ in range(iters):
        Q = P + np.einsum("nij,j->ni", Rins, l); s, Rm, t = umeyama(Q, C); A = s * np.einsum("ij,njk->nik", Rm, Rins).reshape(-1, 3); b = (C - s * (P @ Rm.T) - t).reshape(-1); l = np.linalg.lstsq(A, b, rcond=None)[0]
    Q = P + np.einsum("nij,j->ni", Rins, l); s, Rm, t = umeyama(Q, C); return s, Rm, t, l, ((C - (s * (Q @ Rm.T) + t)) @ Rm) / s, Q
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(MD / "kf_index.json"))}
ts_ins, P_ins, R_ins = traj(MD / "ins_transform.monolithic"); up_axis = int(np.argmin(np.ptp(P_ins, 0)))
z = np.load(R / "experimental/laser_dump/lo_poses.npz"); tl = z["ts_ms"].astype(np.float64); TL = z["T"]; slerp = Slerp(tl, Rotation.from_matrix(TL[:, :3, :3]))
C2L = np.linalg.inv(np.array(json.load(open(R / "prod/monos/rig.json"))["laser_to_camera_left"], np.float64))
def lo_cam(t):
    t = float(np.clip(t, tl[0], tl[-1])); i = np.clip(np.searchsorted(tl, t), 1, len(tl) - 1); w = (t - tl[i - 1]) / max(tl[i] - tl[i - 1], 1e-9)
    M = np.eye(4); M[:3, :3] = slerp([t]).as_matrix()[0]; M[:3, 3] = (1 - w) * TL[i - 1, :3, 3] + w * TL[i, :3, 3]; return (M @ C2L)[:3, 3]
def centres(model):
    ims = read_images_binary(str(model / "images.bin")); return {int(im.name.split("_")[1].split(".")[0]): -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values() if not im.name.endswith("_R.png")}
sources = {}
for label, sub in (("COLMAP after global BA", "aligned"), ("prior (per-block SfM on LO)", "poses")):
    m = PROJ / "camera_calibration" / sub / "sparse/0"
    if (m / "images.bin").exists(): sources[label] = centres(m)
Ks = sorted(set.intersection(*[set(v) for v in sources.values()])); Ks = [k for k in Ks if k in kf]; tq = np.array([kf[k] for k in Ks])
sources["LiDAR odometry (camera centre)"] = {k: lo_cam(kf[k]) for k in Ks}
Pq, Rq = interp(ts_ins, P_ins, R_ins, tq + DT_S * 1e3)
print(f"[traj] {PROJ.name}: {len(Ks)} keyframes; INS at the stamps (+{DT_S} s)", flush=True)
res_all = {}
for label, cen in sources.items():
    C = np.array([cen[k] for k in Ks]); s, Rm, t, l, res, Q = fit_lever(C, Pq, Rq); n = np.linalg.norm(res, axis=1); d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pq, axis=0), axis=1))])
    print(f"[vs INS] {label:30s} scale {s:.4f} (units per INS metre), lever {np.round(l, 2).tolist()} m: residual median {np.median(n):.3f} p90 {np.percentile(n, 90):.3f} max {n.max():.3f} m; quarters {np.round([np.median(n[(d >= a) & (d < b)]) for a, b in ((0, 120), (120, 240), (240, 360), (360, 1e9))], 3).tolist()}", flush=True)
    res_all[label] = (((C - t) @ Rm) / s, res, d)   # the source trajectory expressed in the INS frame (INS metres)
# COLMAP vs LO directly (both in metric-ish frames): how far did the image BA move the cameras off the LiDAR trajectory?
C_ba = np.array([sources["COLMAP after global BA"][k] for k in Ks]); C_lo = np.array([sources["LiDAR odometry (camera centre)"][k] for k in Ks]); C_pr = np.array([sources["prior (per-block SfM on LO)"][k] for k in Ks])
for label, C in (("COLMAP after global BA", C_ba), ("prior (per-block SfM on LO)", C_pr)):
    s, Rm, t = umeyama(C_lo, C); r = np.linalg.norm(C - (s * (C_lo @ Rm.T) + t), axis=1); print(f"[vs LO ] {label:30s} Sim(3) scale {s:.4f}: residual median {np.median(r):.3f} p90 {np.percentile(r, 90):.3f} max {r.max():.3f} (project units)", flush=True)
mv = np.linalg.norm(C_ba - C_pr, axis=1); print(f"[BA    ] camera shift prior -> after global BA: median {np.median(mv):.3f} p90 {np.percentile(mv, 90):.3f} max {mv.max():.3f}", flush=True)
# lane spacings per source (cross-row position of each long lane, INS frame after alignment)
ax_ids = [i for i in range(3) if i != up_axis]; Ph = Pq[:, ax_ids]; e1 = np.linalg.svd(Ph - Ph.mean(0), full_matrices=False)[2][0]; e2 = np.array([-e1[1], e1[0]])
vs = np.convolve(np.gradient(Ph, axis=0) @ e1, np.ones(15) / 15, mode="same"); legs = np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)])
def lanes(Pxy):
    out = []
    for L in np.unique(legs):
        m = legs == L
        if m.sum() < 60 or np.ptp(Ph[m] @ e1) < 30: continue
        core = np.abs((Ph[m] @ e1) - np.median(Ph[m] @ e1)) < 0.35 * np.ptp(Ph[m] @ e1); out.append(float(np.median((Pxy[m][core]) @ e2)))
    return np.array(out)
li = lanes(Ph); print(f"[lanes ] INS spacings {np.round(np.diff(li), 2).tolist()}")
for label, (Pal, res, d) in res_all.items(): ls = lanes(Pal[:, ax_ids]); print(f"[lanes ] {label:30s} {np.round(np.diff(ls), 2).tolist()} (ratio to INS {np.round(np.diff(ls) / np.diff(li), 3).tolist()})")
fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=100); fig.patch.set_facecolor("#1d1a17"); a0, a1 = ax_ids; ax = axs[0]; ax.set_facecolor("#111")
ax.plot(Pq[:, a0], Pq[:, a1], color="#aaa", lw=1.2, label="INS")
for (label, (Pal, res, d)), col in zip(res_all.items(), ("#4fc3f7", "#ff8a65", "#69f0ae")): ax.plot(Pal[:, a0], Pal[:, a1], color=col, lw=0.8, alpha=0.9, label=f"{label} (median {np.median(np.linalg.norm(res, axis=1)):.2f} m vs INS)")
ax.set_aspect("equal"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15); ax.legend(fontsize=9); ax.set_title(f"{PROJ.name}: trajectories Sim(3)+lever-arm aligned onto the INS", color="#e8e0d4")
ax = axs[1]; ax.set_facecolor("#111")
for (label, (Pal, res, d)), col in zip(res_all.items(), ("#4fc3f7", "#ff8a65", "#69f0ae")): ax.plot(d, np.linalg.norm(res, axis=1), lw=0.8, color=col, label=label)
ax.set_yscale("log"); ax.set_xlabel("distance (m)", color="#ccc"); ax.set_ylabel("residual vs INS (m)", color="#ccc"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=9)
out = R / "prod/tassili/ten_rows_sfm_vs_ins_lo.png"; fig.tight_layout(); fig.savefig(out, facecolor=fig.get_facecolor()); print("wrote", out)
