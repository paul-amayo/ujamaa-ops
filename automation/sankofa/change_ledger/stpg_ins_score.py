"""stpg_ins_score.py (2026-10-22) — GNSS-aided INS check of an STPG repeat, the sameday_ins_check.py pair test generalised: for
every localised edge (repeat keyframe k -> teach vertex j) the INS distance between the two instants, both clocks corrected,
in the teach INS frame (repeat INS track aligned to the teach INS track by ICP without any pairing; alignment taken from the
correspondence npz). Also reports the same at clocks 0. The STPG's own prior_dist_m is NOT used: it is a consistency number.
  python3.10 stpg_ins_score.py --stpg <dir/stpg.json> --repeat-name apr1054 --teach-name dec --corr <enu_correspondence_v2 npz> --out <npz>"""
import sys, json, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
from survey_paths import _read_increments
ap = argparse.ArgumentParser(); ap.add_argument("--stpg", required=True); ap.add_argument("--repeat-name", required=True); ap.add_argument("--teach-name", default="dec")
ap.add_argument("--corr", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
C = np.load(a.corr, allow_pickle=True)
def load_ins(path):
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p); xyz = z["xyz"] if "xyz" in z.files else z["xyz_enu"]; return z["ts_ms"].astype(float), np.asarray(xyz, float)
    T = np.eye(4); P = []; ts = []
    for t, m in _read_increments(p):
        m = np.asarray(m, float); T = T @ (m.reshape(4, 4, order="F") if m.ndim == 1 else m); P.append(T[:3, 3].copy()); ts.append(t)
    return np.array(ts, float), np.array(P)
def at(ts, P, q):
    out = np.full((len(q), P.shape[1]), np.nan); ok = (q >= ts[0]) & (q <= ts[-1])
    for d in range(P.shape[1]): out[ok, d] = np.interp(q[ok], ts, P[:, d])
    return out
iT_ts, iT_P = load_ins(str(C["teach_ins"])); iR_ts, iR_P = load_ins(str(C["repeat_ins"])); gT = list(C["teach_ground"]); gR = list(C["repeat_ground"]); R = C["R"]; t = C["t"]
j = json.load(open(a.stpg)); E = j["experiences"]; tsT = np.array(E[a.teach_name]["ts_ms"], float); tsR = np.array(E[a.repeat_name]["ts_ms"], float)
sp_ = [s for s in j["spatial"] if s.get("T_teach_repeat") is not None and s["repeat_exp"] == a.repeat_name]
k = np.array([s["k"] for s in sp_]); jj = np.array([s["j"] for s in sp_]); inl = np.array([s["n_inliers"] for s in sp_])
print(f"[stpg] {a.stpg}: {len(sp_)} localised of {len(tsR)} repeat keyframes; prior of the localised: " + ", ".join(f"{v} {kk}" for kk, v in __import__('collections').Counter(s['chain'] for s in sp_).items()))
res = {}
for name, offT, offR in (("clocks corrected", float(C["teach_offset_ms"]), float(C["repeat_offset_ms"])), ("clocks 0", 0.0, 0.0)):
    pT = at(iT_ts, iT_P, tsT[jj] + offT)[:, gT]; pR = at(iR_ts, iR_P, tsR[k] + offR)[:, gR] @ R.T + t
    ok = np.isfinite(pT).all(1) & np.isfinite(pR).all(1); d = np.linalg.norm(pR[ok] - pT[ok], axis=1)
    print(f"[score] {name} (teach {offT:+.0f} ms, repeat {offR:+.0f} ms): {ok.sum()} pairs; INS distance median {np.median(d):.3f} m, p90 {np.percentile(d, 90):.2f}, "
          f"within 0.5 m {100 * (d <= 0.5).mean():.1f} %, within 1 m {100 * (d <= 1).mean():.1f} %, beyond 3 m {int((d > 3).sum())}, max {d.max():.2f}")
    if name == "clocks corrected":
        for lo, hi in ((12, 20), (20, 40), (40, 10 ** 6)):
            m = (inl[ok] >= lo) & (inl[ok] < hi)
            if m.any(): print(f"         inliers {lo}-{hi if hi < 10**6 else ''}: n {m.sum()}, median {np.median(d[m]):.3f} m, within 0.5 m {100 * (d[m] <= 0.5).mean():.0f} %, beyond 3 m {int((d[m] > 3).sum())}")
    res[name] = dict(k=k[ok], j=jj[ok], dist_m=d, inliers=inl[ok])
np.savez(a.out, **{f"{kk.replace(' ', '_')}_{f}": v for kk, dd in res.items() for f, v in dd.items()}, teach_offset_ms=C["teach_offset_ms"], repeat_offset_ms=C["repeat_offset_ms"])
print("[out]", a.out)
