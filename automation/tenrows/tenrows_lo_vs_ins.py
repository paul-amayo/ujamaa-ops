"""LiDAR odometry (KISS-ICP on the raw laser scans, no IMU, no prior) vs the INS/GNSS trajectory of dec_2025_ten_rows, with
the ZED odometry beside it (Paul, 2026-09-25: "Loam odometry on the laser and compare to ins"). Same machinery as
tenrows_zed_vs_ins2.py: speed-profile clock offset, path-length scale, Sim(3) residuals + drift vs distance, heading
drift, per-lane orientation and lane spacing. Figure -> prod/tassili/ten_rows_lo_vs_ins.png.
  python (any env with numpy/matplotlib) tenrows_lo_vs_ins.py [<lo_poses.npz>]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"
LO = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "experimental/laser_dump/lo_poses.npz"

def traj(path):
    T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in _read_increments(Path(path)):
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m
        T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    ts = np.asarray(ts, np.float64); ts = ts / 1e6 if ts.max() > 1e15 else ts; return ts / 1e3, np.array(P), np.array(Rs)
def umeyama(X, Y, with_scale=True):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = Vt.T @ S @ U.T; s = (np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)) if with_scale else 1.0; return s, Rm, my - s * Rm @ mx
def speed(ts, P, win=1.0):
    v = np.linalg.norm(np.gradient(P, ts, axis=0), axis=1); k = max(1, int(win / np.median(np.diff(ts)))); return np.convolve(v, np.ones(k) / k, mode="same")
def resample(ts_src, X, ts_q): return np.stack([np.interp(ts_q, ts_src, X[:, j]) for j in range(X.shape[1])], 1)
def legs_of(Ph, e1, k=15):
    vs = np.convolve(np.gradient(Ph, axis=0) @ e1, np.ones(k) / k, mode="same"); return np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)]), vs
def leg_table(Ph, legs, vs, e1, e2, min_n, ref, min_len=30.0):
    out = []
    for L in np.unique(legs):
        m = legs == L
        if m.sum() < min_n: continue
        along_ref = ref[m] @ e1
        if np.ptp(along_ref) < min_len: continue
        core = np.abs(along_ref - np.median(along_ref)) < 0.35 * np.ptp(along_ref); Q = Ph[m][core]
        d = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][0]; d = d if d @ e1 > 0 else -d
        out.append((L, "+" if vs[m].mean() > 0 else "-", np.degrees(np.arctan2(d @ e2, d @ e1)), float(np.median(Q @ e2)), float(np.ptp(along_ref))))
    return out

ti, Pi, Ri = traj(MD / "ins_transform.monolithic"); tz, Pz, Rz = traj(MD / "zed_transform_rollfix.monolithic")
z = np.load(LO); tl = z["ts_ms"].astype(np.float64) / 1e3; Pl = z["T"][:, :3, 3]
print(f"[streams] INS {len(ti)} poses path {np.linalg.norm(np.diff(Pi, axis=0), axis=1).sum():.1f} m; LO {len(tl)} poses {tl[-1]-tl[0]:.0f} s path {np.linalg.norm(np.diff(Pl, axis=0), axis=1).sum():.1f} m footprint {np.ptp(Pl, 0).round(1).tolist()}; ZED {len(tz)} poses path {np.linalg.norm(np.diff(Pz, axis=0), axis=1).sum():.1f} m", flush=True)
up_axis = int(np.argmin(np.ptp(Pi, 0))); ax_ids = [i for i in range(3) if i != up_axis]; up = np.zeros(3); up[up_axis] = 1
res_tab = {}; figs = {}
vi = speed(ti, Pi)
for label, (t_s, P_s) in (("LiDAR odometry (KISS-ICP)", (tl, Pl)), ("ZED odometry (roll-fixed)", (tz, Pz))):
    v_s = speed(t_s, P_s); grid = np.arange(-6, 6.001, 0.05); sc = []
    for dt in grid:
        tq = t_s + dt; m = (tq > ti[0] + 2) & (tq < ti[-1] - 2); sc.append(np.corrcoef(np.interp(tq[m], ti, vi), v_s[m])[0, 1] if m.sum() > 100 else np.nan)
    sc = np.array(sc); dt_best = float(grid[np.nanargmax(sc)])
    tq = t_s + dt_best; m = (tq > ti[0]) & (tq < ti[-1]); Pi_q = resample(ti, Pi, tq[m]); Ps = P_s[m]
    Ls = np.linalg.norm(np.diff(Ps, axis=0), axis=1).sum(); Li = np.linalg.norm(np.diff(Pi_q, axis=0), axis=1).sum()
    s, Rm, t = umeyama(Ps, Pi_q); r = Pi_q - (s * (Ps @ Rm.T) + t); n = np.linalg.norm(r, axis=1); d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pi_q, axis=0), axis=1))])
    s1, R1, t1 = umeyama(Ps, Pi_q, False); n1 = np.linalg.norm(Pi_q - (Ps @ R1.T + t1), axis=1)
    q = [np.median(n[(d >= a_) & (d < b_)]) for a_, b_ in ((0, 120), (120, 240), (240, 360), (360, 1e9))]
    print(f"[{label}] clock shift onto INS {dt_best:+.2f} s (speed corr {np.nanmax(sc):.3f}); path {Ls:.1f} vs INS {Li:.1f} m -> units per INS metre {Ls / Li:.4f}; "
          f"Sim(3) scale {s:.4f} residual median {np.median(n):.2f} p90 {np.percentile(n, 90):.2f} max {n.max():.2f} m (by distance quarter {np.round(q, 2).tolist()}); rigid median {np.median(n1):.2f} p90 {np.percentile(n1, 90):.2f}", flush=True)
    Pal = s * (Ps @ Rm.T) + t; res_tab[label] = (Pal, Pi_q, tq[m], n, d)
# heading drift + lanes for both sources against the same INS legs
def heading(P, ts, win=2.0):
    v = np.gradient(P[:, ax_ids], ts, axis=0); k = max(1, int(win / np.median(np.diff(ts)))); v = np.stack([np.convolve(v[:, j], np.ones(k) / k, mode="same") for j in range(2)], 1); return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
tabs = {}
for label, (Pal, Pi_q, tq, n, d) in res_tab.items():
    hi = heading(Pi_q, tq); hs = heading(Pal, tq); mv = speed(tq, Pi_q) > 0.2; dh = np.degrees(hs - hi); dh -= np.median(dh[mv][: max(10, len(dh) // 10)])
    print(f"[{label}] heading drift vs INS: 120 m {np.median(dh[mv & (d > 110) & (d < 130)]):+.1f}, 240 m {np.median(dh[mv & (d > 230) & (d < 250)]):+.1f}, 360 m {np.median(dh[mv & (d > 350) & (d < 370)]):+.1f}, end {np.median(dh[mv & (d > d.max() - 20)]):+.1f} deg; slope {np.polyfit(d[mv], dh[mv], 1)[0] * 100:+.2f} deg per 100 m; vertical range {np.ptp(Pal[:, up_axis]):.2f} m (INS {np.ptp(Pi_q[:, up_axis]):.2f})", flush=True)
    Ph_i = Pi_q[:, ax_ids]; e1 = np.linalg.svd(Ph_i - Ph_i.mean(0), full_matrices=False)[2][0]; legs, vs = legs_of(Ph_i, e1); keep = np.zeros(len(Ph_i), bool)
    for L in np.unique(legs):
        mm = legs == L
        if mm.sum() >= 60 and np.ptp(Ph_i[mm] @ e1) >= 30: keep |= mm
    e1 = np.linalg.svd(Ph_i[keep] - Ph_i[keep].mean(0), full_matrices=False)[2][0]; e2 = np.array([-e1[1], e1[0]]); legs, vs = legs_of(Ph_i, e1)
    ti_tab = leg_table(Ph_i, legs, vs, e1, e2, 60, Ph_i); ts_tab = leg_table(Pal[:, ax_ids], legs, vs, e1, e2, 60, Ph_i); tabs[label] = (ti_tab, ts_tab, dh, mv, d)
    oi = np.array([r[2] for r in ti_tab]); os_ = np.array([r[2] for r in ts_tab]); li = np.array([r[3] for r in ti_tab]); ls = np.array([r[3] for r in ts_tab])
    print(f"[{label}] lanes {len(ti_tab)}: orientation spread INS {np.ptp(oi):.2f} deg vs source {np.ptp(os_):.2f} deg (corr(lane, orient) {np.corrcoef(np.arange(len(os_)), os_)[0, 1]:+.2f}); adjacent-lane spacing INS {np.round(np.diff(li), 2).tolist()} vs source {np.round(np.diff(ls), 2).tolist()} (ratio {np.round(np.diff(ls) / np.diff(li), 3).tolist()})", flush=True)
fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=100); fig.patch.set_facecolor("#1d1a17")
for ax, (label, (Pal, Pi_q, tq, n, d)) in zip(axs[0], res_tab.items()):
    ax.set_facecolor("#111"); ax.plot(Pi_q[:, ax_ids[0]], Pi_q[:, ax_ids[1]], color="#aaa", lw=1.0, label="INS"); ax.plot(Pal[:, ax_ids[0]], Pal[:, ax_ids[1]], color="#ff7043" if "ZED" in label else "#69f0ae", lw=0.9, alpha=0.9, label=label)
    ax.set_aspect("equal"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15); ax.legend(loc="upper right", fontsize=9); ax.set_title(f"{label} (Sim(3) onto INS): residual median {np.median(n):.2f} m", color="#e8e0d4", fontsize=11)
ax = axs[1][0]; ax.set_facecolor("#111")
for label, (ti_tab, ts_tab, dh, mv, d) in tabs.items(): ax.plot(d[mv], dh[mv], lw=0.8, label=label, color="#ff7043" if "ZED" in label else "#69f0ae")
ax.set_xlabel("distance (m)", color="#ccc"); ax.set_ylabel("source − INS heading (deg)", color="#ccc"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.2); ax.legend(fontsize=9); ax.set_title("heading drift vs the INS", color="#e8e0d4", fontsize=11)
ax = axs[1][1]; ax.set_facecolor("#111")
for label, (Pal, Pi_q, tq, n, d) in res_tab.items(): ax.plot(d, n, lw=0.8, label=f"{label}: median {np.median(n):.2f} m", color="#ff7043" if "ZED" in label else "#69f0ae")
ax.set_yscale("log"); ax.set_xlabel("distance (m)", color="#ccc"); ax.set_ylabel("position residual after Sim(3) (m)", color="#ccc"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=9)
out = R / "prod/tassili/ten_rows_lo_vs_ins.png"; fig.tight_layout(); fig.savefig(out, facecolor=fig.get_facecolor()); print("wrote", out)
