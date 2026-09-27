"""Time windows of the ten ten_rows lanes from the KISS-ICP LiDAR odometry (Paul, 2026-09-27: "go for it" — scale the lane-2
recipe to the other rows). The LO track is split into legs by the sign of the velocity along the row axis (dominant
heading of the moving frames, mod 180 deg); legs longer than 30 m are lanes, numbered in time order. Inside a leg the
lane window keeps the straight core: frames moving faster than 0.3 m/s along the axis and within 0.6 m laterally of the
leg's middle-half median, from the first to the last such frame (the turn-in and turn-out are excluded; the camera
looks along the lane so both ends of the tunnel are still seen). Validated against the hand-picked lane-2 window.
Writes experimental/lane_windows.json {lane: {t0_ms, t1_ms, seconds, length_m, lateral_m}}.
  python (h3dgs env) tenrows_lane_windows.py"""
import json
from pathlib import Path
import numpy as np
PAD_S = 1.5   # seconds added at both ends of the straight core (the hand-picked lane 2 ran 1.4 s / 0.8 s further into the turns and trained well)
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental"); z = np.load(R / "laser_dump/lo_poses.npz"); ts = z["ts_ms"].astype(np.float64); P = z["T"][:, :3, 3]
# horizontal plane = the two principal axes of the track (the field is flat); row axis = dominant heading of the moving frames
mu = P.mean(0); U, S, Vt = np.linalg.svd(P - mu, full_matrices=False); xy = (P - mu) @ Vt[:2].T
dt = np.gradient(ts) / 1e3; vel = np.gradient(xy, axis=0) / dt[:, None]; speed = np.linalg.norm(vel, axis=1); moving = speed > 0.3
ang = np.arctan2(vel[moving, 1], vel[moving, 0]); axis_ang = 0.5 * np.arctan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())
e1 = np.array([np.cos(axis_ang), np.sin(axis_ang)]); e2 = np.array([-e1[1], e1[0]]); s = xy @ e1; l = xy @ e2
win = max(3, int(round(1.0 / np.median(dt)))); vs = np.convolve(vel @ e1, np.ones(win) / win, mode="same")
legs = np.concatenate([[0], np.cumsum(np.diff(np.sign(vs)) != 0)]); out = {}; k = 0
for L in np.unique(legs):
    m = np.nonzero(legs == L)[0]
    if len(m) < 30 or np.ptp(s[m]) < 30: continue
    k += 1; mid = m[len(m) // 4: 3 * len(m) // 4]; lat0 = np.median(l[mid])
    core = m[(np.abs(vs[m]) > 0.3) & (np.abs(l[m] - lat0) < 0.6)]
    i0, i1 = core.min(), core.max(); t0, t1 = ts[i0] - PAD_S * 1e3, ts[i1] + PAD_S * 1e3   # pad into the turn-in/out like the hand-picked lane 2 (+1.4 s / +0.8 s)
    out[str(k)] = dict(t0_ms=int(t0), t1_ms=int(t1), seconds=round((t1 - t0) / 1e3, 1), length_m=round(float(abs(s[i1] - s[i0])), 1), lateral_m=round(float(lat0), 2), direction=int(np.sign(vs[m].mean())))
    print(f"lane {k:2d}: {int(t0)}..{int(t1)}  {out[str(k)]['seconds']:5.1f} s  {out[str(k)]['length_m']:5.1f} m along the axis, lateral {lat0:7.2f} m, direction {out[str(k)]['direction']:+d}, ~{int(out[str(k)]['seconds'] * 15)} frames", flush=True)
(R / "lane_windows.json").write_text(json.dumps(out, indent=1))
lat = [out[str(i)]["lateral_m"] for i in range(1, k + 1)]; print(f"{k} lanes; adjacent spacings {np.round(np.abs(np.diff(lat)), 2).tolist()} m -> {R / 'lane_windows.json'}")
known = (1764768208231, 1764768245699); w = out.get("2")
if w: print(f"lane 2 vs the hand-picked window: start {(w['t0_ms'] - known[0]) / 1e3:+.1f} s, end {(w['t1_ms'] - known[1]) / 1e3:+.1f} s")
