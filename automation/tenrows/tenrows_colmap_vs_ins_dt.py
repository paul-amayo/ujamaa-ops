"""Time-offset diagnostic for tenrows_colmap_vs_ins.py: the Sim(3)+lever-arm fit of the survey's COLMAP poses against
the INS trajectory returned a 3.2 m lever arm along the INS body's forward-ish axis — the signature of a constant
stamp offset between INS and images on an out-and-back survey (an along-track error v*dt flips sign leg to leg, exactly
like a forward lever arm). Scan dt, refit, report residual and lever arm per dt; decompose the best fit's residual into
along-leg / cross-leg / vertical components per leg."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"
PROJ = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "experimental/h3dgs_rgb_zed"
def to_ms(ts):
    ts = np.asarray(ts, np.float64); return ts / 1e6 if ts.max() > 1e15 else (ts if ts.max() > 1e11 else ts * 1e3)
def traj(path):
    T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in _read_increments(Path(path)):
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m
        T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    return to_ms(ts), np.array(P), np.array(Rs)
def interp(ts_src, P_src, R_src, ts_q):
    i = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1); t0, t1 = ts_src[i - 1], ts_src[i]
    w = np.clip((ts_q - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]; near = np.where((ts_q - t0) < (t1 - ts_q), i - 1, i)
    return (1 - w) * P_src[i - 1] + w * P_src[i], R_src[near]
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = Vt.T @ S @ U.T; s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X); return s, Rm, my - s * Rm @ mx
def fit(C, P, Rins, lever=True, iters=8):
    l = np.zeros(3)
    for _ in range(iters if lever else 1):
        Q = P + np.einsum("nij,j->ni", Rins, l); s, Rm, t = umeyama(Q, C)
        if not lever: break
        A = s * np.einsum("ij,njk->nik", Rm, Rins).reshape(-1, 3); b = (C - s * (P @ Rm.T) - t).reshape(-1); l = np.linalg.lstsq(A, b, rcond=None)[0]
    Q = P + np.einsum("nij,j->ni", Rins, l); pred = s * (Q @ Rm.T) + t; return s, Rm, t, l, ((C - pred) @ Rm) / s, Q
kf = json.load(open(MD / "kf_index.json")); K = np.array([e["K"] for e in kf]); ts_kf = np.array([float(e["ts_ms"]) for e in kf])
ts_ins, P_ins, R_ins = traj(MD / "ins_transform.monolithic"); up_axis = int(np.argmin(np.ptp(P_ins, 0))); up = np.zeros(3); up[up_axis] = 1
ims = read_images_binary(str(PROJ / "camera_calibration/aligned/sparse/0/images.bin")); cen = {}
for im in ims.values(): cen[int(im.name.split("_")[1].split(".")[0])] = -qvec2rotmat(im.qvec).T @ np.asarray(im.tvec)
idx = np.array([i for i, k in enumerate(K) if int(k) in cen]); C = np.array([cen[int(K[i])] for i in idx]); tq = ts_kf[idx]
print("[dt-scan] dt(s)  scale   |lever| m   lever(body)            3D median  p90     RMS  (Sim3+lever)   | no-lever median")
best = None
for dt in list(np.arange(-8, 8.01, 1.0)) + list(np.arange(-1.5, 1.51, 0.25)):
    Pq, Rq = interp(ts_ins, P_ins, R_ins, tq + dt * 1e3)
    s, Rm, t, l, res, Q = fit(C, Pq, Rq); n3 = np.linalg.norm(res, axis=1); s0, _, _, _, res0, _ = fit(C, Pq, Rq, lever=False); n0 = np.linalg.norm(res0, axis=1)
    print(f"[dt-scan] {dt:+5.2f}  {s:.4f}  {np.linalg.norm(l):7.3f}   {np.round(l, 2).tolist()!s:22}  {np.median(n3):.3f}  {np.percentile(n3, 90):.3f}  {np.sqrt((n3**2).mean()):.3f}          | {np.median(n0):.3f}")
    if best is None or np.median(n0) < best[0]: best = (np.median(n0), dt)
print(f"[dt-scan] best dt without lever arm: {best[1]:+.2f} s (median {best[0]:.3f} m)")
dt = best[1]; Pq, Rq = interp(ts_ins, P_ins, R_ins, tq + dt * 1e3); s, Rm, t, l, res, Q = fit(C, Pq, Rq)
ax_ids = [i for i in range(3) if i != up_axis]; Ph = Pq[:, ax_ids]; e1 = np.linalg.svd(Ph - Ph.mean(0), full_matrices=False)[2][0]
vel = np.gradient(Ph, axis=0) @ e1; vs = np.convolve(vel, np.ones(15) / 15, mode="same"); legs = np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)])
e1_3 = np.zeros(3); e1_3[ax_ids] = e1; e2_3 = np.cross(up, e1_3)
print(f"[best dt {dt:+.2f} s] scale {s:.4f}, lever {np.round(l, 3).tolist()} (|l| {np.linalg.norm(l):.3f} m); per-leg mean residual [along-row, cross-row, vertical] m and median 3D:")
for L in np.unique(legs):
    m = legs == L
    if m.sum() < 30: continue
    r = res[m]; print(f"   leg{L} n={m.sum()} dir {'+' if vs[m].mean() > 0 else '-'}: along {r @ e1_3 @ np.ones(1) / 1 if False else (r @ e1_3).mean():+.2f}  cross {(r @ e2_3).mean():+.2f}  vert {(r @ up).mean():+.2f}  | 3D median {np.median(np.linalg.norm(r, axis=1)):.2f}")
xs = (Q @ e2_3); xc = ((Q + res) @ e2_3); print(f"[best dt] cross-row extent INS {np.ptp(xs):.2f} m vs COLMAP(aligned, source units/scale) {np.ptp(xc):.2f} m; along-row extent INS {np.ptp(Q @ e1_3):.2f} vs COLMAP {np.ptp((Q + res) @ e1_3):.2f} m")
