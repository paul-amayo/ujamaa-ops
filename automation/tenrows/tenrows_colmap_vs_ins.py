"""COLMAP camera poses vs the INS/GNSS (RTK-corrected Fixposition) trajectory of dec_2025_ten_rows, at the survey's
keyframe timestamps (Paul, 2026-09-25: "how does the colmap output compare to the rtk trajectory").
  INS:    prod/monos/monolithics/ins_transform.monolithic (increments -> prefix product; own ENU-like frame),
          interpolated to the keyframes' timestamps (kf_index.json ts_ms).
  COLMAP: <h3dgs proj>/camera_calibration/aligned/sparse/0/images.bin = the survey's globally bundle-adjusted poses
          (per-block GLOMAP refine, fixed camera, then global BA); also the prior model (before the global BA) and the
          roll-fixed ZED odometry (zed_transform_rollfix.monolithic, the pose source before any image refinement).
  Fit per source: Sim(3) Umeyama (scale free) + INS->camera lever arm (alternating least squares; the INS sits
          elsewhere on the vehicle and the rows are driven out-and-back, so a lever arm shows as a heading-dependent
          offset). Reports scale, residuals (3D / horizontal / vertical: median, p90, max, RMS), per-leg medians and
          mean offsets, residual vs distance (drift trend); figure -> prod/tassili/ten_rows_colmap_vs_ins.png.
  python3.10 (nerf_new) tenrows_colmap_vs_ins.py [<h3dgs proj dir>]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"
PROJ = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "experimental/h3dgs_rgb_zed"

def to_ms(ts):
    ts = np.asarray(ts, np.float64); return ts / 1e6 if ts.max() > 1e15 else (ts if ts.max() > 1e11 else ts * 1e3)

def traj(path):
    """Increment stream -> (ts_ms [N], positions [N,3], rotations [N,3,3]); first record = identity."""
    incs = _read_increments(Path(path)); T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in incs:
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m
        T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    return to_ms(ts), np.array(P), np.array(Rs)

def interp(ts_src, P_src, R_src, ts_q):
    i = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1); t0, t1 = ts_src[i - 1], ts_src[i]
    w = np.clip((ts_q - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)[:, None]
    near = np.where((ts_q - t0) < (t1 - ts_q), i - 1, i)
    return (1 - w) * P_src[i - 1] + w * P_src[i], R_src[near], np.minimum(np.abs(ts_q - t0), np.abs(t1 - ts_q))

def umeyama(X, Y, with_scale=True):
    """Sim(3) s,R,t with Y ~ s R X + t (least squares)."""
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = Vt.T @ S @ U.T; s = (np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)) if with_scale else 1.0
    return s, Rm, my - s * Rm @ mx

def fit(C, P, Rins, lever=True, with_scale=True, iters=8):
    """Camera centres C ~ s R (P + Rins l) + t. Returns dict with s, R, t, l, residual vectors (in the INS frame, metres)."""
    l = np.zeros(3)
    for _ in range(iters if lever else 1):
        Q = P + np.einsum("nij,j->ni", Rins, l); s, Rm, t = umeyama(Q, C, with_scale)
        if not lever: break
        A = s * np.einsum("ij,njk->nik", Rm, Rins).reshape(-1, 3); b = (C - s * (P @ Rm.T) - t).reshape(-1)
        l = np.linalg.lstsq(A, b, rcond=None)[0]
    Q = P + np.einsum("nij,j->ni", Rins, l); pred = s * (Q @ Rm.T) + t
    res_ins = ((C - pred) @ Rm) / s        # residual expressed in the INS frame, in INS metres
    return {"s": s, "R": Rm, "t": t, "l": l, "res": res_ins, "pred": pred, "Q": Q}

def stats(res, up):
    n3 = np.linalg.norm(res, axis=1); h = np.linalg.norm(res - np.outer(res @ up, up), axis=1); v = res @ up
    return (f"3D median {np.median(n3):.3f} p90 {np.percentile(n3, 90):.3f} max {n3.max():.3f} RMS {np.sqrt((n3 ** 2).mean()):.3f} m | "
            f"horizontal median {np.median(h):.3f} p90 {np.percentile(h, 90):.3f} | vertical median |{np.median(np.abs(v)):.3f}| p90 {np.percentile(np.abs(v), 90):.3f} m")

import os
DT_S = float(os.environ.get("DT_S", "0"))   # constant stamp offset applied to the keyframe times before the INS lookup (tenrows_colmap_vs_ins_dt.py scan: +3 s)
kf = json.load(open(MD / "kf_index.json")); K = np.array([e["K"] for e in kf]); ts_kf = np.array([float(e["ts_ms"]) for e in kf]) + DT_S * 1e3
print(f"[cfg] keyframe stamps shifted by {DT_S:+.2f} s before the INS lookup", flush=True)
ts_ins, P_ins, R_ins = traj(MD / "ins_transform.monolithic")
print(f"[ins] {len(ts_ins)} poses, {(ts_ins[-1] - ts_ins[0]) / 1e3:.0f} s, path {np.linalg.norm(np.diff(P_ins, axis=0), axis=1).sum():.1f} m, footprint {np.ptp(P_ins, 0).round(1).tolist()} m; "
      f"keyframes {len(kf)} spanning {(ts_kf[-1] - ts_kf[0]) / 1e3:.0f} s (INS covers {ts_ins[0] <= ts_kf[0]} / {ts_ins[-1] >= ts_kf[-1]})", flush=True)
up_axis = int(np.argmin(np.ptp(P_ins, 0))); up = np.zeros(3); up[up_axis] = 1.0   # the flat axis of the INS trajectory
Pq, Rq, dt = interp(ts_ins, P_ins, R_ins, ts_kf); print(f"[ins] interpolation gap to nearest INS sample: median {np.median(dt):.1f} ms, max {dt.max():.1f} ms")
# cumulative INS distance at the keyframes and leg segmentation from the along-row direction sign
d_ins = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pq, axis=0), axis=1))])
ax_ids = [i for i in range(3) if i != up_axis]; Ph = Pq[:, ax_ids]; e1 = np.linalg.svd(Ph - Ph.mean(0), full_matrices=False)[2][0]
vel = np.gradient(Ph, axis=0) @ e1; k = 15; vs = np.convolve(vel, np.ones(k) / k, mode="same"); sign = np.sign(vs)
legs = np.concatenate([[0], np.cumsum(np.diff(sign) != 0)])
sources = {}
def colmap_centres(model_dir):
    ims = read_images_binary(str(model_dir / "images.bin")); out = {}
    for im in ims.values():
        Rw = qvec2rotmat(im.qvec); out[int(im.name.split("_")[1].split(".")[0])] = -Rw.T @ np.asarray(im.tvec)
    return out
for label, mdir in (("COLMAP aligned (global BA)", PROJ / "camera_calibration/aligned/sparse/0"), ("COLMAP prior (per-block refine, pre-BA)", PROJ / "camera_calibration/prior/sparse/0")):
    if (mdir / "images.bin").exists(): sources[label] = colmap_centres(mdir)
for label, mono in (("ZED odometry (roll-fixed)", "zed_transform_rollfix.monolithic"), ("ZED odometry + INS heading correction", "zed_transform_rollfix_inscorr.monolithic")):
    if (MD / mono).exists():
        t, P, Rr = traj(MD / mono); Pk, _, _ = interp(t, P, Rr, ts_kf); sources[label] = {int(k): Pk[i] for i, k in enumerate(K)}
results = {}
for label, cen in sources.items():
    idx = np.array([i for i, k in enumerate(K) if int(k) in cen]); C = np.array([cen[int(K[i])] for i in idx])
    r0 = fit(C, Pq[idx], Rq[idx], lever=False); r1 = fit(C, Pq[idx], Rq[idx], lever=True); results[label] = (idx, r1)
    print(f"\n[{label}] {len(idx)} keyframes; Sim(3) scale (source units per INS metre) {r0['s']:.4f} -> with lever arm {r1['s']:.4f}; lever arm (INS frame) {np.round(r1['l'], 3).tolist()} m (|l| {np.linalg.norm(r1['l']):.3f})")
    print(f"   rigid+scale only : {stats(r0['res'], up)}")
    print(f"   + lever arm      : {stats(r1['res'], up)}")
    rf = fit(C, Pq[idx], Rq[idx], lever=True, with_scale=False); print(f"   scale fixed at 1 : {stats(rf['res'], up)}")
    n3 = np.linalg.norm(r1["res"], axis=1); dd = d_ins[idx]; trend = np.polyfit(dd, n3, 1)[0] * 100
    print(f"   residual vs distance travelled: slope {trend:+.2f} m per 100 m; first-quarter median {np.median(n3[dd < dd.max() / 4]):.3f} vs last-quarter {np.median(n3[dd > 3 * dd.max() / 4]):.3f} m")
    lg = legs[idx]; per = []
    for L in np.unique(lg):
        m = lg == L
        if m.sum() < 30: continue
        per.append(f"leg{L}:{np.median(n3[m]):.2f}({np.round(r1['res'][m].mean(0), 2).tolist()})")
    print(f"   per-leg median residual (mean offset vector, INS frame): {' '.join(per)}")
# figure: top-down INS vs aligned COLMAP (global BA) + residual vs distance for every source
lab0 = "COLMAP aligned (global BA)" if "COLMAP aligned (global BA)" in results else list(results)[0]; idx, r = results[lab0]
fig, axs = plt.subplots(1, 2, figsize=(20, 9), dpi=110); fig.patch.set_facecolor("#1d1a17")
axp = axs[0]; axp.set_facecolor("#111"); a0, a1 = ax_ids
axp.plot(P_ins[:, a0], P_ins[:, a1], color="#888", lw=0.8, label="INS / RTK trajectory")
res_m = np.linalg.norm(r["res"], axis=1); Cq = r["Q"] + r["res"]
sc = axp.scatter(Cq[:, a0], Cq[:, a1], c=res_m, cmap="magma", s=9, vmin=0, vmax=max(0.2, np.percentile(res_m, 95)), label=f"{lab0} (aligned)")
axp.quiver(r["Q"][:, a0], r["Q"][:, a1], r["res"][:, a0] * 10, r["res"][:, a1] * 10, color="#4fc3f7", scale=1, scale_units="xy", angles="xy", width=0.0015, alpha=0.6, label="residual ×10")
axp.set_aspect("equal"); axp.tick_params(colors="#ccc"); axp.grid(alpha=0.15); axp.legend(loc="upper right", fontsize=8)
axp.set_title(f"{lab0} vs INS: scale {r['s']:.4f}, lever {np.linalg.norm(r['l']):.2f} m, 3D median {np.median(res_m):.3f} m", color="#e8e0d4", fontsize=11)
cb = fig.colorbar(sc, ax=axp, fraction=0.03); cb.set_label("residual (m)", color="#ccc"); cb.ax.tick_params(colors="#ccc")
axr = axs[1]; axr.set_facecolor("#111")
for label, (i2, r2) in results.items(): axr.plot(d_ins[i2], np.linalg.norm(r2["res"], axis=1), lw=0.9, label=f"{label}: median {np.median(np.linalg.norm(r2['res'], axis=1)):.3f} m")
for b in np.where(np.diff(legs) != 0)[0]: axr.axvline(d_ins[b], color="#555", lw=0.5)
axr.set_yscale("log"); axr.set_xlabel("distance travelled along the INS trajectory (m)", color="#ccc"); axr.set_ylabel("3D residual after Sim(3)+lever-arm alignment (m)", color="#ccc")
axr.tick_params(colors="#ccc"); axr.grid(alpha=0.2, which="both"); axr.legend(fontsize=8); axr.set_title("residual vs distance (leg boundaries in grey)", color="#e8e0d4", fontsize=11)
out = R / "prod/tassili/ten_rows_colmap_vs_ins.png"; fig.tight_layout(); fig.savefig(out, facecolor=fig.get_facecolor()); print("\nwrote", out)
