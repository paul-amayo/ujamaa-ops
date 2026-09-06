"""Refine ZED odometry with INS heading: estimate the ZED yaw drift vs the INS
(velocity headings, time-aligned), apply it as a per-increment yaw correction
(dead reckoning with corrected heading), conjugate to the laser frame with the
rig L2C, write a new pose monolithic for the projector. No INS->lidar extrinsic
needed: only the INS heading-over-time is used (constant offset removed).
Run WITHOUT the C++ binding in the process (pbTransform_pb2 clash)."""
from PIL import Image  # noqa (PIL first)
import sys, json, numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from survey_paths import _read_increments
import ensure_laser_frame_poses as elf          # read_transforms / write_transforms (pb2), FLAT_RATIO etc.
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
M = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics"); R = M.parent; T = R.parent/"tassili"
L2C = np.array(json.load(open(R/"rig.json"))["laser_to_camera_left"], float); L2Ci = np.linalg.inv(L2C)

def load(path):
    incs = _read_increments(path); ts = np.array([s for s, _ in incs], float)
    mats = [m.reshape(4, 4, order="F") for _, m in incs]
    dest = np.r_[ts[1:], ts[-1] + (ts[-1] - ts[-2])]           # dest ts of record i = source ts of record i+1
    return ts, dest, mats
def integrate(mats):
    Tm = np.eye(4); out = []
    for m in mats: Tm = Tm @ m; out.append(Tm.copy())
    return np.array(out)
def heading(P, t, flat, win_s=2.0):
    ax = [i for i in range(3) if i != flat]; Q = P[:, ax]
    dt = np.median(np.diff(t)) / 1000.0; k = max(3, int(win_s / dt))
    V = np.gradient(uniform_filter1d(Q, k, axis=0), t / 1000.0, axis=0)
    sp = np.linalg.norm(V, axis=1); h = np.unwrap(np.arctan2(V[:, 1], V[:, 0])); return h, sp

import os
ZED_IN = os.environ.get("ZED_IN", "zed_transform.monolithic"); SUF = os.environ.get("OUT_SUFFIX", "inscorr")
zts, zdest, zm = load(M/ZED_IN); its, idest, im = load(M/"ins_transform.monolithic")
Z = integrate(zm); I = integrate(im); Pz, Pi = Z[:, :3, 3], I[:, :3, 3]
fz = int(np.argmin(np.ptp(Pz, 0))); fi = int(np.argmin(np.ptp(Pi, 0)))
print(f"ZED {len(zm)} poses flat axis {'xyz'[fz]} | INS {len(im)} poses flat axis {'xyz'[fi]} | time overlap {max(zdest[0], idest[0]):.0f}..{min(zdest[-1], idest[-1]):.0f} ms")
def nanfix(x, t, name):
    bad = ~np.isfinite(x); print(f"  {name}: {bad.sum()} non-finite of {len(x)}")
    if bad.any(): x = x.copy(); x[bad] = np.interp(t[bad], t[~bad], x[~bad])
    return x
def heading_nan(P, t, flat):
    ax = [i for i in range(3) if i != flat]; Q = P[:, ax]
    dt = np.median(np.diff(t)) / 1000.0; k = max(3, int(2.0 / dt))
    V = np.gradient(uniform_filter1d(Q, k, axis=0), t / 1000.0, axis=0)
    sp = np.linalg.norm(V, axis=1); ang = np.arctan2(V[:, 1], V[:, 0])
    return ang, sp
az, spz = heading_nan(Pz, zdest, fz); ai, spi = heading_nan(Pi, idest, fi)
az = nanfix(az, zdest, "ZED heading"); ai = nanfix(ai, idest, "INS heading"); spz = nanfix(spz, zdest, "ZED speed"); spi = nanfix(spi, idest, "INS speed")
hz = np.unwrap(az); hi = np.unwrap(ai)
print(f"  unwrapped: ZED {np.isfinite(hz).all()} INS {np.isfinite(hi).all()}; ZED heading range {np.degrees(np.ptp(hz)):.0f} deg, INS {np.degrees(np.ptp(hi)):.0f} deg")
# ---- leg-matched drift estimate (robust: instantaneous heading rates only correlate at r~-0.4 here) ----
def legs_time(P3, t):
    flat = int(np.argmin(np.ptp(P3, axis=0))); ax = [i for i in range(3) if i != flat]; P = P3[:, ax]
    pc = P - P.mean(0); e1 = np.linalg.svd(pc, full_matrices=False)[2][0]; sc = pc @ e1; k = max(5, len(sc) // 160)
    ss = uniform_filter1d(sc, k); ds = np.sign(np.gradient(ss)); turns = np.where(np.diff(ds) != 0)[0]
    b = [turns[0]] if len(turns) else []
    for tt in turns[1:]:
        if tt - b[-1] > len(sc) // 65: b.append(tt)
    bounds = [0] + b + [len(P) - 1]; out = []
    for i in range(len(bounds) - 1):
        a, z = bounds[i], bounds[i + 1]
        if z - a < len(sc) // 40: continue
        d = P[z] - P[a]
        if np.linalg.norm(d) < 20: continue
        ang = np.degrees(np.arctan2(d @ np.array([-e1[1], e1[0]]), d @ e1)); ang = ((ang + 90) % 180) - 90
        out.append((0.5 * (t[a] + t[z]), ang, t[a], t[z]))
    return out
LZ = legs_time(Pz, zdest); LI = legs_time(Pi, idest)
print(f"  legs: ZED {len(LZ)}, INS {len(LI)}")
# match legs by time overlap
pairs = []
for tm, az_, a0, a1 in LZ:
    cand = [(abs(tm - ti), ai_) for ti, ai_, b0, b1 in LI if min(a1, b1) - max(a0, b0) > 0.5 * (a1 - a0)]
    if cand: pairs.append((tm, az_, min(cand)[1]))
tmid = np.array([p[0] for p in pairs]); thz = np.array([p[1] for p in pairs]); thi = np.array([p[2] for p in pairs])
best_s, best_spread = None, 1e9
for sgn_m in (+1.0, -1.0):                      # mirror between INS ground plane and camera-world
    dd = thz - sgn_m * thi; dd = dd - np.median(dd); spr = np.ptp(dd)
    print(f"  mirror sign {sgn_m:+.0f}: per-leg (ZED - s*INS) after const removal: {np.round(dd,1)} (spread {spr:.1f} deg)")
    if spr < best_spread: best_spread, best_s = spr, sgn_m
d_leg = (thz - best_s * thi); d_leg = d_leg - d_leg[0]           # drift relative to the first leg (deg)
print(f"  using mirror sign {best_s:+.0f}; leg drift (deg): {np.round(d_leg,1)}")
# drift applied to ZED heading must be -(ZED - INS) so corrected ZED == INS (up to const)
drift_sm = np.radians(-np.interp(zdest, tmid, d_leg))
print(f"  drift correction: start {np.degrees(drift_sm[0]):+.1f} deg, end {np.degrees(drift_sm[-1]):+.1f} deg, span {np.degrees(np.ptp(drift_sm)):.1f} deg")
moving = spz > 0.15; drift = drift_sm; hi_at_z = hz + drift_sm   # for diagnostics/plot compatibility
ddrift = np.diff(np.r_[drift_sm[0], drift_sm])

def legs_of(P3):
    flat = int(np.argmin(np.ptp(P3, axis=0))); ax = [i for i in range(3) if i != flat]; P = P3[:, ax]
    pc = P - P.mean(0); e1 = np.linalg.svd(pc, full_matrices=False)[2][0]; s = pc @ e1; k = max(5, len(s) // 160)
    ss = uniform_filter1d(s, k); ds = np.sign(np.gradient(ss)); turns = np.where(np.diff(ds) != 0)[0]
    b = [turns[0]] if len(turns) else []
    for t in turns[1:]:
        if t - b[-1] > len(s) // 65: b.append(t)
    bounds = [0] + b + [len(P) - 1]; out = []
    for i in range(len(bounds) - 1):
        a, z = bounds[i], bounds[i + 1]
        if z - a < len(s) // 40: continue
        d = P[z] - P[a]
        if np.linalg.norm(d) < 20: continue
        ang = np.degrees(np.arctan2(d @ np.array([-e1[1], e1[0]]), d @ e1)); out.append(((ang + 90) % 180) - 90)
    return np.array(out)
def spread_trend(P3):
    a = legs_of(P3); return len(a), np.ptp(a), (np.corrcoef(np.arange(len(a)), a)[0, 1] if len(a) > 2 else 0)
print(f"  legs   ZED raw: n={spread_trend(Pz)[0]} spread {spread_trend(Pz)[1]:.1f} deg corr {spread_trend(Pz)[2]:+.2f} | INS: n={spread_trend(Pi)[0]} spread {spread_trend(Pi)[1]:.1f} corr {spread_trend(Pi)[2]:+.2f}")
# apply the correction with both yaw signs about the camera vertical (flat axis fz); keep the better
axis = np.zeros(3); axis[fz] = 1.0
best = None
for sgn in (+1.0, -1.0):
    mats_c = [m @ np.block([[Rot.from_rotvec(sgn * dd * axis).as_matrix(), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]) for m, dd in zip(zm, ddrift)]
    Zc = integrate(mats_c); n, spr, cr = spread_trend(Zc[:, :3, 3])
    hzc, _ = heading(Zc[:, :3, 3], zdest, fz); res = np.degrees(np.std(np.unwrap(hi_at_z - hzc)[moving]))
    print(f"  sign {sgn:+.0f}: corrected legs n={n} spread {spr:.1f} deg corr {cr:+.2f} | heading residual vs INS std {res:.2f} deg | footprint {np.round(np.ptp(Zc[:, :3, 3], 0)[[i for i in range(3) if i != fz]], 1)} m")
    if best is None or res < best[0]: best = (res, sgn, mats_c, Zc)
res, sgn, mats_c, Zc = best; print(f"  -> using sign {sgn:+.0f}")
# regression check: with zero correction, my conjugation must match the tool's laserframe file
if ZED_IN == "zed_transform.monolithic":
    tool = elf.read_transforms(M/"transform_lio_laserframe.monolithic")
    mine0 = [L2Ci @ m @ L2C for m in zm]; print(f"  regression: my L2C^-1*inc*L2C vs tool's laserframe stream: max|diff| {max(np.abs(a - b[1]).max() for a, b in zip(mine0, tool)):.2e}")
# write corrected camera-frame stream (provenance) and its laser-frame conjugation (for the projector)
src_msgs = [m for m, _ in elf.read_transforms(M/ZED_IN)]
elf.write_transforms(src_msgs, mats_c, M/f"zed_transform_{SUF}.monolithic")
elf.write_transforms(src_msgs, [L2Ci @ m @ L2C for m in mats_c], M/f"transform_lio_laserframe_{SUF}.monolithic")
for f in (f"zed_transform_{SUF}.monolithic", f"transform_lio_laserframe_{SUF}.monolithic"):
    idx = M/(f + ".index"); idx.unlink(missing_ok=True); print(f"  wrote {f} ({(M/f).stat().st_size} B)")
np.savez_compressed(T/f"zed_{SUF}_diag.npz", t=zdest, drift=drift_sm, raw=drift, Pz=Pz, Pzc=Zc[:, :3, 3], Pi=Pi, ti=idest)
fig, axs = plt.subplots(1, 3, figsize=(21, 6), dpi=110); fig.patch.set_facecolor("#1d1a17")
axs[0].set_facecolor("#111"); axs[0].plot((tmid - zdest[0]) / 1000, -d_leg, "o", ms=6, color="#7fd", label="per-leg (INS - ZED)")
axs[0].plot((zdest - zdest[0]) / 1000, np.degrees(drift_sm), color="#ff5533", lw=2, label="smoothed, applied"); axs[0].set_xlabel("time (s)", color="#ccc"); axs[0].set_ylabel("INS − ZED heading (deg)", color="#ccc"); axs[0].legend(fontsize=8); axs[0].tick_params(colors="#ccc"); axs[0].grid(alpha=0.15); axs[0].set_title("estimated ZED yaw drift", color="#e8e0d4")
for ax, P3, ttl in ((axs[1], Pz, "ZED raw"), (axs[2], Zc[:, :3, 3], f"ZED corrected with INS (sign {sgn:+.0f})")):
    f = int(np.argmin(np.ptp(P3, 0))); a_ = [i for i in range(3) if i != f]; ax.set_facecolor("#111")
    ax.plot(P3[:, a_[0]], P3[:, a_[1]], color="#7fd", lw=1); ax.set_aspect("equal"); ax.set_title(f"{ttl}: legs spread {spread_trend(P3)[1]:.1f} deg", color="#e8e0d4"); ax.tick_params(colors="#ccc"); ax.grid(alpha=0.15)
fig.suptitle("ten_rows: ZED odometry refined with INS heading", color="#e8e0d4", fontsize=13); fig.tight_layout(); fig.savefig(T/"ten_rows_zed_inscorr.png", facecolor=fig.get_facecolor()); print("  wrote ten_rows_zed_inscorr.png")
