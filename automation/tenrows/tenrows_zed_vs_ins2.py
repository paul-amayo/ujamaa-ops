"""ZED visual odometry vs the INS/GNSS trajectory of dec_2025_ten_rows at full rate (Paul, 2026-09-25: "compare zed
odometry to the ins data"). Streams: zed_transform_rollfix.monolithic (15 Hz, the roll-corrected ZED odometry the survey
poses descend from), zed_transform_rollfix_inscorr.monolithic (the same with per-leg INS heading correction),
ins_transform.monolithic (30 Hz Fixposition INS). Frame-invariant checks first (speed-profile cross-correlation for the
clock offset, path length for scale), then Sim(3) alignment residuals, drift vs distance, heading drift, per-leg
orientation and lateral position (lane spacing) for each map. Figure -> prod/tassili/ten_rows_zed_vs_ins2.png.
  python3.10 (nerf_new) tenrows_zed_vs_ins2.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"

def traj(path):
    T = np.eye(4); ts, P, Rs = [], [], []
    for t, m in _read_increments(Path(path)):
        m = np.asarray(m, np.float64); m = m.reshape(4, 4, order="F") if m.ndim == 1 else m
        T = T @ m; ts.append(t); P.append(T[:3, 3].copy()); Rs.append(T[:3, :3].copy())
    ts = np.asarray(ts, np.float64); ts = ts / 1e6 if ts.max() > 1e15 else ts
    return ts / 1e3, np.array(P), np.array(Rs)          # seconds

def umeyama(X, Y, with_scale=True):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = Vt.T @ S @ U.T; s = (np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)) if with_scale else 1.0
    return s, Rm, my - s * Rm @ mx

def speed(ts, P, win=1.0):
    v = np.linalg.norm(np.gradient(P, ts, axis=0), axis=1); k = max(1, int(win / np.median(np.diff(ts)))); return np.convolve(v, np.ones(k) / k, mode="same")

def resample(ts_src, X, ts_q):
    return np.stack([np.interp(ts_q, ts_src, X[:, j]) for j in range(X.shape[1])], 1)

def legs_of(Ph, e1, k=15):
    vs = np.convolve(np.gradient(Ph, axis=0) @ e1, np.ones(k) / k, mode="same"); return np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)]), vs

def leg_table(Ph, legs, vs, e1, e2, min_n, min_len=30.0, ref=None):
    """Per leg (only legs spanning >= min_len m along the rows in the reference map): direction, orientation of the
    core (middle 70 %, away from the turns), lateral position, along-row length."""
    out = []
    for L in np.unique(legs):
        m = legs == L
        if m.sum() < min_n: continue
        along_ref = (ref if ref is not None else Ph)[m] @ e1
        if np.ptp(along_ref) < min_len: continue
        along = Ph[m] @ e1; core = np.abs(along_ref - np.median(along_ref)) < 0.35 * np.ptp(along_ref)   # middle 70 %: away from the turns
        Q = Ph[m][core]; d = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][0]; d = d if d @ e1 > 0 else -d
        out.append((L, "+" if vs[m].mean() > 0 else "-", np.degrees(np.arctan2(d @ e2, d @ e1)), float(np.median(Q @ e2)), float(np.ptp(along))))
    return out

ti, Pi, Ri = traj(MD / "ins_transform.monolithic")
tz, Pz, Rz = traj(MD / "zed_transform_rollfix.monolithic")
tc, Pc, Rc = traj(MD / "zed_transform_rollfix_inscorr.monolithic")
print(f"[streams] INS {len(ti)} poses {ti[-1]-ti[0]:.0f} s path {np.linalg.norm(np.diff(Pi, axis=0), axis=1).sum():.1f} m; ZED {len(tz)} poses {tz[-1]-tz[0]:.0f} s path {np.linalg.norm(np.diff(Pz, axis=0), axis=1).sum():.1f} m; "
      f"ZED starts {tz[0]-ti[0]:+.1f} s vs INS start (header clocks)", flush=True)
# 1. clock offset from the speed profiles (frame-invariant): shift ZED time by dt so that INS(t) matches ZED(t + dt)
vi, vz = speed(ti, Pi), speed(tz, Pz); grid = np.arange(-6, 6.001, 0.05); scores = []
for dt in grid:
    tq = tz + dt; m = (tq > ti[0] + 2) & (tq < ti[-1] - 2)
    a = np.interp(tq[m], ti, vi); b = vz[m]; scores.append(np.corrcoef(a, b)[0, 1] if m.sum() > 100 else np.nan)
scores = np.array(scores); dt_best = float(grid[np.nanargmax(scores)])
print(f"[clock] speed-profile cross-correlation: best ZED->INS shift {dt_best:+.2f} s (corr {np.nanmax(scores):.3f}; at 0 s {scores[np.argmin(np.abs(grid))]:.3f}); stamp headers said INS leads the cameras by +2.385 s", flush=True)
# 2. scale from path length over the common window
tq = tz + dt_best; m = (tq > ti[0]) & (tq < ti[-1]); Pi_q = resample(ti, Pi, tq[m]); Ri_q = np.array([Ri[np.searchsorted(ti, t).clip(0, len(ti) - 1)] for t in tq[m]])
Lz = np.linalg.norm(np.diff(Pz[m], axis=0), axis=1).sum(); Li = np.linalg.norm(np.diff(Pi_q, axis=0), axis=1).sum()
print(f"[scale] common window {tq[m][-1]-tq[m][0]:.0f} s: ZED path {Lz:.1f} vs INS path {Li:.1f} m -> ZED units per INS metre {Lz / Li:.4f}", flush=True)
# 3. Sim(3) and rigid alignment of the ZED map onto the INS (every ZED sample), residuals and drift
res_tab = {}
Pc_m = Pc[m] if (len(tc) == len(tz) and np.allclose(tc, tz)) else resample(tc, Pc, tz[m])   # the corrected stream lives on the ZED clock: sample it at the ZED times, not the shifted ones
for label, P in (("ZED roll-fixed", Pz[m]), ("ZED + INS heading correction", Pc_m)):
    for ws in (True, False):
        s, Rm, t = umeyama(P, Pi_q, ws); r = Pi_q - (s * (P @ Rm.T) + t); n = np.linalg.norm(r, axis=1); d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pi_q, axis=0), axis=1))])
        q = [np.median(n[(d >= a) & (d < b)]) for a, b in ((0, 120), (120, 240), (240, 360), (360, 1e9))]
        print(f"[align] {label:30s} {'Sim(3)':6s} scale {s:.4f}: residual median {np.median(n):.2f} p90 {np.percentile(n, 90):.2f} max {n.max():.2f} m; by distance quarter {np.round(q, 2).tolist()}" if ws else
              f"[align] {label:30s} {'rigid':6s}             residual median {np.median(n):.2f} p90 {np.percentile(n, 90):.2f} max {n.max():.2f} m", flush=True)
        if ws: res_tab[label] = (s, Rm, t, P, n, d)
# 4. heading drift: horizontal velocity heading of each map after alignment, unwrapped, difference vs distance
up_axis = int(np.argmin(np.ptp(Pi, 0))); ax_ids = [i for i in range(3) if i != up_axis]; up = np.zeros(3); up[up_axis] = 1
def heading(P, ts, win=2.0):
    v = np.gradient(P[:, ax_ids], ts, axis=0); k = max(1, int(win / np.median(np.diff(ts)))); v = np.stack([np.convolve(v[:, j], np.ones(k) / k, mode="same") for j in range(2)], 1); return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
s, Rm, t, Pzz, n, d = res_tab["ZED roll-fixed"]; Pz_al = s * (Pzz @ Rm.T) + t
hi = heading(Pi_q, tq[m]); hz = heading(Pz_al, tq[m]); mv = speed(tq[m], Pi_q) > 0.2; dh = np.degrees(hz - hi); dh -= np.median(dh[mv][: len(dh) // 10])   # zero at the start
print(f"[heading] ZED-INS heading difference (moving samples, zeroed at the start): after 120 m {np.median(dh[mv & (d > 110) & (d < 130)]):+.1f} deg, 240 m {np.median(dh[mv & (d > 230) & (d < 250)]):+.1f}, 360 m {np.median(dh[mv & (d > 350) & (d < 370)]):+.1f}, end {np.median(dh[mv & (d > d.max() - 20)]):+.1f} deg; "
      f"drift rate {np.polyfit(d[mv], dh[mv], 1)[0] * 100:+.2f} deg per 100 m", flush=True)
# 5. legs: orientation and lateral position per leg in INS, ZED (aligned), ZED corrected (aligned)
Ph_i = Pi_q[:, ax_ids]; e1 = np.linalg.svd(Ph_i - Ph_i.mean(0), full_matrices=False)[2][0]; e2 = np.array([-e1[1], e1[0]])
legs, vs = legs_of(Ph_i, e1); min_n = 60
keep = np.zeros(len(Ph_i), bool)   # second pass: row axis from the long legs only (the diagonal entry run and the turns bias the first PCA)
for L in np.unique(legs):
    mm = legs == L
    if mm.sum() >= min_n and np.ptp(Ph_i[mm] @ e1) >= 30.0: keep |= mm
e1 = np.linalg.svd(Ph_i[keep] - Ph_i[keep].mean(0), full_matrices=False)[2][0]; e2 = np.array([-e1[1], e1[0]]); legs, vs = legs_of(Ph_i, e1)
ti_tab = leg_table(Ph_i, legs, vs, e1, e2, min_n)
tz_tab = leg_table(Pz_al[:, ax_ids], legs, vs, e1, e2, min_n, ref=Ph_i)
s2, R2, t2, Pcc, _, _ = res_tab["ZED + INS heading correction"]; Pc_al = s2 * (Pcc @ R2.T) + t2; tc_tab = leg_table(Pc_al[:, ax_ids], legs, vs, e1, e2, min_n, ref=Ph_i)
print("[legs] leg dir | orientation (deg): INS  ZED  ZED+corr | lateral position (m): INS  ZED  ZED+corr | leg length INS")
for a, b, c in zip(ti_tab, tz_tab, tc_tab):
    print(f"[legs] {a[0]:3d} {a[1]}  | {a[2]:+7.2f} {b[2]:+7.2f} {c[2]:+7.2f} | {a[3]:7.2f} {b[3]:7.2f} {c[3]:7.2f} | {a[4]:6.1f}")
oi = np.array([r[2] for r in ti_tab]); oz = np.array([r[2] for r in tz_tab]); oc = np.array([r[2] for r in tc_tab]); di = np.array([r[1] == "+" for r in ti_tab])
print(f"[legs] orientation spread: INS {np.ptp(oi):.2f} deg (outbound mean {oi[di].mean():+.2f}, return mean {oi[~di].mean():+.2f}), ZED {np.ptp(oz):.2f} (corr(leg, orient) {np.corrcoef(np.arange(len(oz)), oz)[0, 1]:+.2f}), ZED+corr {np.ptp(oc):.2f}")
li = np.array([r[3] for r in ti_tab]); lz = np.array([r[3] for r in tz_tab]); lc = np.array([r[3] for r in tc_tab])
print(f"[legs] adjacent-lane spacing INS      {np.round(np.diff(li), 2).tolist()}\n[legs] adjacent-lane spacing ZED      {np.round(np.diff(lz), 2).tolist()}\n[legs] adjacent-lane spacing ZED+corr {np.round(np.diff(lc), 2).tolist()}")
# 6. vertical: INS up-axis profile vs ZED (aligned) up-axis profile
zi = Pi_q[:, up_axis]; zz = Pz_al[:, up_axis]; print(f"[vertical] INS range {np.ptp(zi):.2f} m, ZED(aligned) range {np.ptp(zz):.2f} m, difference RMS {np.sqrt(((zz - zi) - (zz - zi).mean()) ** 2).mean():.2f} m")
# figure
fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=100); fig.patch.set_facecolor("#1d1a17")
for ax, (label, Pal) in zip(axs[0], (("ZED roll-fixed (Sim(3) onto INS)", Pz_al), ("ZED + INS heading correction (Sim(3) onto INS)", Pc_al))):
    ax.set_facecolor("#111"); ax.plot(Pi_q[:, ax_ids[0]], Pi_q[:, ax_ids[1]], color="#aaa", lw=1.0, label="INS")
    ax.plot(Pal[:, ax_ids[0]], Pal[:, ax_ids[1]], color="#ff7043", lw=0.9, alpha=0.9, label=label)
    for r in ti_tab: ax.annotate(f"{r[0]}", (r[3] * e2 + (Ph_i.mean(0) @ e1) * e1)[[0, 1]] if False else (Ph_i[legs == r[0]].mean(0)), color="#ffe082", fontsize=9, ha="center")
    ax.set_aspect("equal"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15); ax.legend(loc="upper right", fontsize=9); ax.set_title(label, color="#e8e0d4", fontsize=11)
ax = axs[1][0]; ax.set_facecolor("#111"); ax.plot(d[mv], dh[mv], color="#4fc3f7", lw=0.8); ax.set_xlabel("distance (m)", color="#ccc"); ax.set_ylabel("ZED − INS heading (deg)", color="#ccc"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.2)
for b in np.where(np.diff(legs) != 0)[0]: ax.axvline(d[b], color="#555", lw=0.5)
ax.set_title("heading drift of the ZED odometry vs the INS (velocity headings, 2 s smoothing)", color="#e8e0d4", fontsize=11)
ax = axs[1][1]; ax.set_facecolor("#111")
for label, (s_, R_, t_, P_, n_, d_) in res_tab.items(): ax.plot(d_, n_, lw=0.8, label=f"{label}: median {np.median(n_):.2f} m")
ax.set_yscale("log"); ax.set_xlabel("distance (m)", color="#ccc"); ax.set_ylabel("position residual after Sim(3) (m)", color="#ccc"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=9)
ax.set_title(f"clock offset from speed profiles {dt_best:+.2f} s; ZED units per INS metre {Lz / Li:.4f}", color="#e8e0d4", fontsize=11)
out = R / "prod/tassili/ten_rows_zed_vs_ins2.png"; fig.tight_layout(); fig.savefig(out, facecolor=fig.get_facecolor()); print("wrote", out)
