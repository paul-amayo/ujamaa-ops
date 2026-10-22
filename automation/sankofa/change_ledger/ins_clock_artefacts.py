"""ins_clock_artefacts.py (2026-10-22) — machine-readable record of the Klapmuts A300 image-vs-INS clock offsets and the corrected same-day
STPG check (numbers as in lab_notebook/2026-10.md, 2026-10-22). Writes stpg_dec_apr_20261021/ins_clock_offsets.json and
apr1130_vs_1054/ins_check_pairs.npz. Offset convention: INS stamp = camera/lidar stamp + offset for the same instant, so sample the INS at
(image ts + offset). system python3 with aru_sil_core/build/datatypes on the path."""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
from survey_paths import _read_increments
K = Path('/home/paperspace/data/klapmuts'); G = K / 'dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021'
def traj(mono):
    T = np.eye(4); P = []; ts = []
    for t, m in _read_increments(Path(mono)):
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
def own_scan(P, ts, iTs, iP, offs):
    out = []
    for off in offs:
        Q = at(iTs, iP, ts + off); g = np.isfinite(P).all(1) & np.isfinite(Q).all(1); R, t = rigid2(P[g], Q[g]); r = np.linalg.norm(P[g] @ R.T + t - Q[g], axis=1)
        out.append((int(off), float(np.median(r)), float(np.percentile(r, 90)), int(g.sum())))
    return out
j = json.load(open(G / 'apr1130_vs_1054/stpg.json')); E = j['experiences']
tsT, cT = np.array(E['apr1130']['ts_ms'], float), np.array(E['apr1130']['c2w'], float); tsR, cR = np.array(E['apr1054']['ts_ms'], float), np.array(E['apr1054']['c2w'], float)
iT_ts, iT_P = traj(K / 'apr_2026_1130/prod/monos/monolithics/ins_transform.monolithic'); iT_P = iT_P[:, [0, 2]]          # camera convention: ground = x, z
z = np.load(K / 'apr_2026_zed/experimental/ins/ins_odom.npz'); iR_ts, iR_P = z['ts_ms'].astype(float), z['xyz'][:, :2]     # FP_ENU0: ground = east, north
d = np.load(G / 'lo_superpoint/dec.npz', allow_pickle=True); zd = np.load(K / 'dec_2025_ten_rows/experimental/ins_enu_abs.npz'); dts, dP = d['ts_ms'].astype(float), d['c2w'][:, [0, 2], 3]
rec = {'convention': 'INS stamp = camera/lidar stamp + offset_ms for the same instant; sample INS at (image ts + offset_ms)', 'method': 'rigid 2-D fit of the pass\'s own keyframe pose stream to its INS track over an offset scan (10 ms steps near the minimum); the 11:39 offset from the same-day STPG pairs because its KISS-ICP track is broken at the lidar gap', 'passes': {}}
for name, P, ts, iTs, iP, note in [('apr_2026_zed (April 10:54)', cR[:, [0, 2], 3], tsR, iR_ts, iR_P, 'pose stream = prod/monos/transform_lio (July ingest); INS experimental/ins/ins_odom.npz'),
                                    ('dec_2025_ten_rows (December)', dP, dts, zd['ts_ms'].astype(float), zd['xyz'][:, :2], 'pose stream = KISS-ICP SuperPoint teach lo_superpoint/dec.npz; INS experimental/ins_enu_abs.npz')]:
    coarse = own_scan(P, ts, iTs, iP, range(-12000, 12001, 100)); c0 = min(coarse, key=lambda r: r[1]); fine = own_scan(P, ts, iTs, iP, range(c0[0] - 300, c0[0] + 301, 10)); b = min(fine, key=lambda r: r[1]); z0 = [r for r in coarse if r[0] == 0][0]
    flat = [r[0] for r in fine if r[1] <= b[1] + 0.01]
    rec['passes'][name] = {'offset_ms': b[0], 'offset_range_within_1cm_ms': [flat[0], flat[-1]], 'median_m_at_offset': round(b[1], 3), 'p90_m_at_offset': round(b[2], 3), 'median_m_at_0': round(z0[1], 2), 'keyframes': b[3], 'source': note}
    print(f"[offset] {name}: {b[0]} ms ({flat[0]}..{flat[-1]}), median {b[1]:.3f} m at the offset vs {z0[1]:.2f} m at 0")
# same-day pairs: INS-to-INS track ICP (no pairing), then the INS distance of every STPG pair at 0 / corrected offsets
tree = cKDTree(iT_P); best = None
for y0 in np.radians(np.arange(-180, 180, 10)):
    R = np.array([[np.cos(y0), -np.sin(y0)], [np.sin(y0), np.cos(y0)]]); t = iT_P.mean(0) - iR_P.mean(0) @ R.T
    for _ in range(60):
        dd, ii = tree.query(iR_P @ R.T + t); keep = dd <= np.quantile(dd, 0.8); R, t = rigid2(iR_P[keep], iT_P[ii[keep]])
    m = float(np.median(tree.query(iR_P @ R.T + t)[0]))
    if best is None or m < best[2]: best = (R, t, m)
R, t, nn = best; sp = [s for s in j['spatial'] if s.get('T_teach_repeat') is not None]; k = np.array([s['k'] for s in sp]); jj = np.array([s['j'] for s in sp])
offA = rec['passes']['apr_2026_zed (April 10:54)']['offset_ms']; offT_scan = []
for offT in range(offA - 300, offA + 301, 10):
    pR = at(iR_ts, iR_P, tsR[k] + offA); pT = at(iT_ts, iT_P, tsT[jj] + offT); ok = np.isfinite(pR).all(1) & np.isfinite(pT).all(1); offT_scan.append((offT, float(np.median(np.linalg.norm(pR[ok] @ R.T + t - pT[ok], axis=1)))))
bT = min(offT_scan, key=lambda r: r[1]); flatT = [r[0] for r in offT_scan if r[1] <= bT[1] + 0.01]
rec['passes']['apr_2026_1130 (April 11:39)'] = {'offset_ms': bT[0], 'offset_range_within_1cm_ms': [flatT[0], flatT[-1]], 'source': 'from the same-day STPG pairs with the 10:54 offset fixed; its own KISS-ICP keyframes vs its INS: 1.53 m one rigid piece, 0.17 / 0.16 m when split at ~328 s (the 3.6 s lidar gap)'}
res = {}; arrays = {}
for tag, oR, oT in [('clocks_0', 0, 0), ('clocks_corrected', offA, bT[0])]:
    pR = at(iR_ts, iR_P, tsR[k] + oR); pT = at(iT_ts, iT_P, tsT[jj] + oT); ok = np.isfinite(pR).all(1) & np.isfinite(pT).all(1); e = pR[ok] @ R.T + t - pT[ok]; n = np.linalg.norm(e, axis=1)
    res[tag] = {'offsets_ms': {'10:54': oR, '11:39': oT}, 'pairs': int(ok.sum()), 'median_m': round(float(np.median(n)), 3), 'p90_m': round(float(np.percentile(n, 90)), 2), 'within_0p5m': round(float((n <= 0.5).mean()), 3), 'within_1m': round(float((n <= 1).mean()), 3), 'over_3m': int((n > 3).sum()), 'max_m': round(float(n.max()), 2)}
    arrays[tag] = (k[ok], jj[ok], n, e)
rec['same_day_check'] = {'what': 'GNSS-aided INS distance between each 10:54 keyframe and the 11:39 keyframe the STPG matched it to; INS frames aligned by track ICP without any pairing', 'ins_track_icp_median_nn_m': round(nn, 3), **res, 'stpg_camera_offset_median_m': round(float(np.median([np.linalg.norm(np.array(s['T_teach_repeat'])[[0, 2], 3]) for s in sp])), 3)}
json.dump(rec, open(G / 'ins_clock_offsets.json', 'w'), indent=1)
kc, jc, nc, ec = arrays['clocks_corrected']; k0, j0, n0, e0 = arrays['clocks_0']
np.savez(G / 'apr1130_vs_1054/ins_check_pairs.npz', k=kc, j=jc, dist_m=nc, err_xy=ec, k_clock0=k0, j_clock0=j0, dist_m_clock0=n0, offsets_ms=np.array([offA, bT[0]]), note='k = 10:54 keyframe, j = 11:39 teach vertex; dist = INS distance at the corrected offsets (10:54, 11:39); *_clock0 = with both INS clocks at 0')
print(json.dumps(rec['same_day_check'], indent=0)[:900]); print('[wrote]', G / 'ins_clock_offsets.json', G / 'apr1130_vs_1054/ins_check_pairs.npz')
