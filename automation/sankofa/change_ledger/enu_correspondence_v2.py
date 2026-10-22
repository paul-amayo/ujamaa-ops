"""enu_correspondence_v2.py (2026-10-22) — Dec→Apr keyframe correspondence from the RAW INS tracks with the image-vs-INS
clock offsets corrected (A100 session, 22 Oct: INS stamp = camera stamp + offset; December 2,510 ms, April 6,960 ms; sample
the INS at image ts + offset). Replaces enu_correspondence.py (21 Oct), which gave December no offset and chose April's
from {0, ±5000} by keyframe coverage. Inputs: the teach's vertex stamps from its STPG experience, the repeat's keyframe
stamps from survey_paths.kf_timestamps, the INS tracks as npz (ts_ms, xyz) or as a composed ins_transform monolithic.
Output npz: jstar[k] (teach vertex for repeat keyframe k, -1 none), dist[k], the track alignment, both INS tracks.
  python3.10 enu_correspondence_v2.py --teach-stpg <lo_superpoint/stpg.json> --teach-name dec --teach-ins <npz> --teach-ground 0,1
      --teach-offset-ms 2510 --repeat-root <survey> --repeat-ins <npz|monolithic> --repeat-ground 0,1 --repeat-offset-ms 6960 --out <npz>"""
import sys, json, argparse, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
import survey_paths as sp
from survey_paths import _read_increments

ap = argparse.ArgumentParser()
ap.add_argument("--teach-stpg", required=True); ap.add_argument("--teach-name", default="dec")
ap.add_argument("--teach-ins", required=True); ap.add_argument("--teach-ground", default="0,1"); ap.add_argument("--teach-offset-ms", type=float, required=True)
ap.add_argument("--repeat-root", required=True); ap.add_argument("--repeat-ins", required=True); ap.add_argument("--repeat-ground", default="0,1")
ap.add_argument("--repeat-offset-ms", type=float, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--radius", type=float, default=1.5, help="max INS distance for a correspondence")
a = ap.parse_args()

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
def rigid2(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca
def icp2(A, B, it=60, trim=0.8):
    """trimmed NN ICP over a yaw sweep and a mirror check, A -> B, from the tracks alone (no pairing)"""
    tree = cKDTree(B); best = None
    for flip in (1, -1):
        Af = A * np.array([1, flip])
        for y0 in np.radians(np.arange(-180, 180, 10)):
            R = np.array([[np.cos(y0), -np.sin(y0)], [np.sin(y0), np.cos(y0)]]); t = B.mean(0) - Af.mean(0) @ R.T
            for _ in range(it):
                Ap = Af @ R.T + t; d, i = tree.query(Ap); keep = d <= np.quantile(d, trim); R, t = rigid2(Af[keep], B[i[keep]])
            d, _ = tree.query(Af @ R.T + t); m = float(np.median(d))
            if best is None or m < best[3]: best = (R @ np.diag([1, flip]), t, flip, m)
    return best

gT = [int(x) for x in a.teach_ground.split(",")]; gR = [int(x) for x in a.repeat_ground.split(",")]
E = json.load(open(a.teach_stpg))["experiences"][a.teach_name]; tsT = np.array(E["ts_ms"], float)
tsR = np.asarray(sp.kf_timestamps(a.repeat_root), float)
iT_ts, iT_P = load_ins(a.teach_ins); iR_ts, iR_P = load_ins(a.repeat_ins)
print(f"[in] teach {a.teach_name}: {len(tsT)} vertices; INS {len(iT_P)} poses, ground axes {gT}, offset {a.teach_offset_ms:+.0f} ms")
print(f"[in] repeat {Path(a.repeat_root).name}: {len(tsR)} keyframes; INS {len(iR_P)} poses, ground axes {gR}, offset {a.repeat_offset_ms:+.0f} ms")
PT = at(iT_ts, iT_P, tsT + a.teach_offset_ms)[:, gT]; PR = at(iR_ts, iR_P, tsR + a.repeat_offset_ms)[:, gR]
okT = np.isfinite(PT).all(1); okR = np.isfinite(PR).all(1)
print(f"[in] INS covers {okT.sum()}/{len(tsT)} teach vertices and {okR.sum()}/{len(tsR)} repeat keyframes at the corrected clocks")
# align the repeat INS frame to the teach INS frame from the two tracks (no pairing)
R, t, flip, nn = icp2(iR_P[:, gR], iT_P[:, gT])
yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0])); print(f"[align] repeat INS -> teach INS: yaw {yaw:.2f} deg, t {np.round(t, 2).tolist()}, mirror {flip == -1}, median NN {nn:.3f} m")
PR2 = PR @ R.T + t
# headings from the INS tracks (central differences on the sampled positions)
def heading(P, ok):
    H = np.full_like(P, np.nan); idx = np.flatnonzero(ok)
    v = np.gradient(P[idx], axis=0); n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9; H[idx] = v / n; return H
HT = heading(PT, okT); HR = heading(PR2, okR)
tree = cKDTree(PT[okT]); idxT = np.flatnonzero(okT)
jstar = np.full(len(tsR), -1); dist = np.full(len(tsR), np.nan); same = opp = 0
for k in np.flatnonzero(okR):
    d, i = tree.query(PR2[k], k=8); cos = (HT[idxT[i]] * HR[k]).sum(1); g = (cos > 0.5) & (d <= a.radius)
    if g.any(): jstar[k] = idxT[i[g][0]]; dist[k] = d[g][0]
    same += bool(((d <= 1.0) & (cos > 0.5)).any()); opp += bool(((d <= 1.0) & (cos < -0.5)).any())
n = int(okR.sum()); found = int((jstar >= 0).sum())
print(f"[corr] repeat keyframes {n}: same-heading teach vertex within 1 m {same} ({100 * same / n:.0f} %), opposite within 1 m {opp} ({100 * opp / n:.0f} %); "
      f"correspondence (radius {a.radius} m) for {found} ({100 * found / len(tsR):.0f} %), INS distance median {np.nanmedian(dist):.3f} m, p90 {np.nanpercentile(dist, 90):.2f} m")
np.savez(a.out, jstar=jstar, dist=dist, yaw_deg=yaw, t=t, mirror=(flip == -1), teach_offset_ms=a.teach_offset_ms, repeat_offset_ms=a.repeat_offset_ms,
         PT=PT, PR_in_teach=PR2, tsT=tsT, tsR=tsR, teach_ground=gT, repeat_ground=gR, R=R, teach_ins=str(a.teach_ins), repeat_ins=str(a.repeat_ins))
print("[out]", a.out)
