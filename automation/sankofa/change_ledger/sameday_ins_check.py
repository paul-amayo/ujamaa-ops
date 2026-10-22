"""sameday_ins_check.py (2026-10-22) — independent GNSS check of the Klapmuts same-day STPG pair (10:54 repeat vs 11:39 teach, stpg_dec_apr_20261021/apr1130_vs_1054)
against both passes' Fixposition INS tracks: INS-to-INS track ICP (no pairing), INS distance between matched keyframes, image-vs-INS clock-offset scans,
along/across-row split, the 11:39 KISS-ICP break at the lidar gap; writes ins_check_clock.png. system python3 (aru_sil_core build/datatypes on the path)."""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
from survey_paths import _read_increments
K = Path('/home/paperspace/data/klapmuts'); G = K / 'dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/apr1130_vs_1054/stpg.json'
j = json.load(open(G)); E = j['experiences']
tsT, cT = np.array(E['apr1130']['ts_ms'], float), np.array(E['apr1130']['c2w'], float); tsR, cR = np.array(E['apr1054']['ts_ms'], float), np.array(E['apr1054']['c2w'], float)
def traj(mono):
    T = np.eye(4); P = []; ts = []
    for t, m in _read_increments(Path(mono)):
        m = np.asarray(m, float); T = T @ (m.reshape(4, 4, order="F") if m.ndim == 1 else m); P.append(T[:3, 3].copy()); ts.append(t)
    return np.array(ts, float), np.array(P)
iT_ts, iT_P = traj(K / 'apr_2026_1130/prod/monos/monolithics/ins_transform.monolithic')
z = np.load(K / 'apr_2026_zed/experimental/ins/ins_odom.npz'); iR_ts, iR_P = z['ts_ms'].astype(float), z['xyz']
print(f"[ins] 11:39 INS {len(iT_P)} poses, ts {iT_ts[0]:.0f}..{iT_ts[-1]:.0f} (keyframes {tsT[1]:.0f}..{tsT[-1]:.0f}); extent {np.ptp(iT_P, 0).round(1).tolist()}")
print(f"[ins] 10:54 INS {len(iR_P)} poses (FP_ENU0), ts {iR_ts[0]:.0f}..{iR_ts[-1]:.0f} (keyframes {tsR[0]:.0f}..{tsR[-1]:.0f}); extent {np.ptp(iR_P, 0).round(1).tolist()}")
def at(ts, P, q):  # linear interpolation of positions at times q (nan outside)
    out = np.full((len(q), 3), np.nan); ok = (q >= ts[0]) & (q <= ts[-1])
    for d in range(3): out[ok, d] = np.interp(q[ok], ts, P[:, d])
    return out
def rigid2(A, B):  # 2-D rigid A -> B
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca
def icp2(A, B, it=60, trim=0.8):  # trimmed NN ICP over a yaw sweep (and mirror check), A -> B, independent of any pairing
    tree = cKDTree(B); best = None
    for flip in (1, -1):
      Af = A * np.array([1, flip])
      for y0 in np.radians(np.arange(-180, 180, 10)):
        R = np.array([[np.cos(y0), -np.sin(y0)], [np.sin(y0), np.cos(y0)]]); t = B.mean(0) - Af.mean(0) @ R.T
        for _ in range(it):
            Ap = Af @ R.T + t; d, i = tree.query(Ap); keep = d <= np.quantile(d, trim); R, t = rigid2(Af[keep], B[i[keep]])
        d, _ = tree.query(Af @ R.T + t); m = float(np.median(d))
        if best is None or m < best[3]: best = (R @ np.diag([1, flip]), t, flip, m)
    print(f'[icp] best: mirror {best[2] == -1}, median NN {best[3]:.2f} m')
    return best[0], best[1], best[3]
def _unused(A, B, it=60, trim=0.8):
    tree = cKDTree(B); R, t = np.eye(2), B.mean(0) - A.mean(0)
    for _ in range(it):
        Ap = A @ R.T + t; d, i = tree.query(Ap); keep = d <= np.quantile(d, trim); R, t = rigid2(A[keep], B[i[keep]])
    d, _ = tree.query(A @ R.T + t); return R, t, float(np.median(d))
sp = [s for s in j['spatial'] if s.get('T_teach_repeat') is not None]; k = np.array([s['k'] for s in sp]); jj = np.array([s['j'] for s in sp]); ttr = np.array([np.linalg.norm(np.array(s['T_teach_repeat'])[[0, 2], 3]) for s in sp])
for name, offR, offT in [('INS clocks as recorded in the 10:54 file (-5000 ms) / 0', -5000, 0), ('both 0 ms', 0, 0)]:
    pR = at(iR_ts, iR_P, tsR[k] + offR)[:, :2]; pT = at(iT_ts, iT_P, tsT[jj] + offT)[:, [0, 2]]; ok = np.isfinite(pR).all(1) & np.isfinite(pT).all(1)
    # align the two INS frames from the TRACKS alone (no STPG pairing): 10:54 INS track -> 11:39 INS track
    R, t, nn = icp2(iR_P[:, :2], iT_P[:, [0, 2]])
    e = (pR[ok] @ R.T + t) - pT[ok]; en = np.linalg.norm(e, axis=1)
    print(f"[check] {name}: {ok.sum()} pairs | track ICP yaw {np.degrees(np.arctan2(R[1, 0], R[0, 0])):.2f} deg, median NN {nn:.2f} m | INS distance between the two matched keyframes: median {np.median(en):.2f} m, p90 {np.percentile(en, 90):.2f}, max {en.max():.2f}; within 0.5 m {100 * (en <= 0.5).mean():.0f} %, within 1 m {100 * (en <= 1).mean():.0f} %, > 3 m {100 * (en > 3).mean():.0f} % | camera offset |t| median {np.median(ttr[ok]):.2f} m")
    # the same with the best rigid fit ON the pairs (removes any frame misfit; what remains is non-rigid)
    R2, t2 = rigid2(pR[ok], pT[ok]); e2 = np.linalg.norm((pR[ok] @ R2.T + t2) - pT[ok], axis=1)
    print(f"         after a rigid fit on the pairs themselves: median {np.median(e2):.2f} m, p90 {np.percentile(e2, 90):.2f}, max {e2.max():.2f}")
    np.savez('/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad/sameday_ins_' + str(offR) + '.npz', k=k[ok], j=jj[ok], e=e, e2=e2, tR=tsR[k[ok]], pR=pR[ok] @ R.T + t, pT=pT[ok])

# ---- clock-offset scan and along / cross-track decomposition ----
R, t, nn = icp2(iR_P[:, :2], iT_P[:, [0, 2]])
def resid(offR, offT):
    pR = at(iR_ts, iR_P, tsR[k] + offR)[:, :2]; pT = at(iT_ts, iT_P, tsT[jj] + offT)[:, [0, 2]]; ok = np.isfinite(pR).all(1) & np.isfinite(pT).all(1)
    return ok, (pR @ R.T + t) - pT
best = None
for offT in range(-3000, 3001, 100):
    ok, e = resid(0, offT); m = np.median(np.linalg.norm(e[ok], axis=1))
    if best is None or m < best[1]: best = (offT, m)
print(f"[clock] 11:39 image-INS offset scan (10:54 at 0): best {best[0]} ms -> median {best[1]:.2f} m")
best2 = None
for offR in range(-3000, 3001, 100):
    ok, e = resid(offR, 0); m = np.median(np.linalg.norm(e[ok], axis=1))
    if best2 is None or m < best2[1]: best2 = (offR, m)
print(f"[clock] 10:54 image-INS offset scan (11:39 at 0): best {best2[0]} ms -> median {best2[1]:.2f} m")
for offR, offT in [(0, 0), (0, best[0]), (best2[0], 0)]:
    ok, e = resid(offR, offT); e = e[ok]; kk, jx = k[ok], jj[ok]
    # along-track direction of the 11:39 INS track at the teach vertex
    pT_a = at(iT_ts, iT_P, tsT[jx] + offT - 300)[:, [0, 2]]; pT_b = at(iT_ts, iT_P, tsT[jx] + offT + 300)[:, [0, 2]]; d = pT_b - pT_a; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    al = (e * d).sum(1); cr = e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]; n = np.linalg.norm(e, axis=1)
    h, edges = np.histogram(al, bins=np.arange(-4.25, 4.26, 0.5))
    print(f"[decomp] offsets 10:54 {offR} ms, 11:39 {offT} ms: |e| median {np.median(n):.2f} m p90 {np.percentile(n, 90):.2f} | along-track median {np.median(al):+.2f} m, |along| p90 {np.percentile(np.abs(al), 90):.2f} | cross-track median {np.median(cr):+.2f} m, |cross| p90 {np.percentile(np.abs(cr), 90):.2f}, |cross| > 1.2 m: {100 * (np.abs(cr) > 1.2).mean():.1f} %")
    print('          along-track histogram (0.5 m bins from -4.25):', h.tolist())

# ---- structure of the along-track error: time series, speed, per-pair discontinuities ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
offR, offT = 0, 0
ok, e = resid(offR, offT); e = e[ok]; kk, jx = k[ok], jj[ok]
pT_a = at(iT_ts, iT_P, tsT[jx] - 300)[:, [0, 2]]; pT_b = at(iT_ts, iT_P, tsT[jx] + 300)[:, [0, 2]]; dv = pT_b - pT_a; sp_T = np.linalg.norm(dv, axis=1) / 0.6; d = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
pR_a = at(iR_ts, iR_P, tsR[kk] - 300)[:, :2]; pR_b = at(iR_ts, iR_P, tsR[kk] + 300)[:, :2]; sp_R = np.linalg.norm(pR_b - pR_a, axis=1) / 0.6
al = (e * d).sum(1); tR = (tsR[kk] - tsR[0]) / 1000
# does the matched teach vertex index move smoothly? jumps in j between consecutive localised keyframes
dj = np.diff(jx); print(f"[j] teach vertex steps between consecutive repeat keyframes: median {np.median(dj):.0f}, backwards {(dj < 0).sum()}, jumps > 5 vertices {(np.abs(dj) > 5).sum()}")
c1 = np.corrcoef(al, sp_T - sp_R)[0, 1]; c2 = np.corrcoef(al, sp_T)[0, 1]; c3 = np.corrcoef(al, sp_R)[0, 1]
print(f"[speed] along-track error vs speed: corr with 11:39 speed {c2:+.2f}, with 10:54 speed {c3:+.2f}, with the speed difference {c1:+.2f}; speeds median 11:39 {np.median(sp_T):.2f} m/s, 10:54 {np.median(sp_R):.2f} m/s")
# the 11:39 teach's own KISS-ICP keyframe track vs its INS at the same stamps (local consistency of the teach frame)
cT2 = cT[jx][:, [0, 2], 3]; pTi = at(iT_ts, iT_P, tsT[jx])[:, [0, 2]]; g = np.isfinite(cT2).all(1) & np.isfinite(pTi).all(1)
Rk, tk = rigid2(cT2[g], pTi[g]); rk = np.linalg.norm(cT2[g] @ Rk.T + tk - pTi[g], axis=1)
print(f"[teach] 11:39 KISS-ICP keyframe positions vs its own INS at the same image stamps, rigid fit: median {np.median(rk):.2f} m, p90 {np.percentile(rk, 90):.2f}, max {rk.max():.2f}")
fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True); fig.subplots_adjust(0.08, 0.07, 0.99, 0.95, 0, 0.12)
ax[0].plot(tR, al, '.', ms=2, color='#2F5D8A'); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_ylabel('along-row error, m'); ax[0].set_ylim(-5, 5); ax[0].set_title('Same-day pair: 10:54 keyframe vs the 11:39 keyframe the STPG matched it to, distance by the INS (both clocks 0 ms)', fontsize=9)
ax[1].plot(tR, (e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]), '.', ms=2, color='#1E8C5A'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_ylabel('across-row error, m'); ax[1].set_ylim(-3, 3)
ax[2].plot(tR, sp_R, '-', lw=0.8, color='#B3261E', label='10:54 speed'); ax[2].plot(tR, sp_T, '-', lw=0.8, color='#C9A227', label='11:39 speed at the matched vertex'); ax[2].set_ylabel('speed, m/s'); ax[2].set_xlabel('time into the 10:54 pass, s'); ax[2].legend(fontsize=8, loc='upper right')
out = '/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/apr1130_vs_1054/ins_check_alongrow.png'; fig.savefig(out, dpi=150); print('[fig]', out)

# ---- absolute image-vs-INS clock: each pass's own pose stream (keyframe c2w at image stamps) vs its INS, rigid fit, offset scan ----
def own_scan(c2w, ts, iTs, iP, ground_ins, lo=-10000, hi=10001, step=100):
    P = c2w[:, [0, 2], 3]; g0 = np.isfinite(P).all(1); res = []
    for off in range(lo, hi, step):
        Q = at(iTs, iP, ts + off)[:, ground_ins]; g = g0 & np.isfinite(Q).all(1)
        Rr, tt = rigid2(P[g], Q[g]); r = np.linalg.norm(P[g] @ Rr.T + tt - Q[g], axis=1); res.append((off, float(np.median(r)), float(np.percentile(r, 90)), int(g.sum())))
    return res
for name, c2w, ts, iTs, iP, gi in [('11:39 KISS-ICP keyframes vs 11:39 INS', cT, tsT, iT_ts, iT_P, [0, 2]), ('10:54 pose stream keyframes vs 10:54 INS', cR, tsR, iR_ts, iR_P, [0, 1])]:
    res = own_scan(c2w, ts, iTs, iP, gi); b = min(res, key=lambda r: r[1]); z0 = [r for r in res if r[0] == 0][0]
    print(f"[own] {name}: at 0 ms median {z0[1]:.2f} m (p90 {z0[2]:.2f}); best offset {b[0]:+d} ms -> median {b[1]:.2f} m, p90 {b[2]:.2f} ({b[3]} keyframes)")
    print('      scan (offset ms: median m):', ' '.join(f"{r[0]}:{r[1]:.1f}" for r in res[::10]))

# ---- the pair test with the 10:54 INS clock fixed at +7000 ms and the 11:39 one scanned (its own-pose scan is broad, 5-8 s) ----
print('[pairs] 10:54 INS +7000 ms; 11:39 INS offset scan:')
best = None
for offT in range(4000, 10001, 100):
    ok, e = resid(7000, offT); n = np.linalg.norm(e[ok], axis=1); m = float(np.median(n))
    if best is None or m < best[1]: best = (offT, m, float(np.percentile(n, 90)), float((n <= 0.5).mean()), float((n <= 1).mean()), float(n.max()), int(ok.sum()))
    if offT % 1000 == 0: print(f'   {offT} ms: median {m:.2f} m')
print(f"[pairs] best 11:39 offset {best[0]} ms: median {best[1]:.2f} m, p90 {best[2]:.2f}, within 0.5 m {100 * best[3]:.0f} %, within 1 m {100 * best[4]:.0f} %, max {best[5]:.2f} ({best[6]} pairs)")
ok, e = resid(7000, best[0]); e = e[ok]; jx = jj[ok]
pT_a = at(iT_ts, iT_P, tsT[jx] + best[0] - 300)[:, [0, 2]]; pT_b = at(iT_ts, iT_P, tsT[jx] + best[0] + 300)[:, [0, 2]]; dv = pT_b - pT_a; d = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
al = (e * d).sum(1); cr = e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]
print(f"[pairs] along-row median {np.median(al):+.2f} m, |along| p90 {np.percentile(np.abs(al), 90):.2f}; across-row median {np.median(cr):+.2f} m, |across| p90 {np.percentile(np.abs(cr), 90):.2f}; pairs > 1 m apart: {(np.linalg.norm(e, axis=1) > 1).sum()}, > 3 m: {(np.linalg.norm(e, axis=1) > 3).sum()}")
# same check of the 11:39 teach's own KISS-ICP frame at that offset (what the teach frame carries)
P = cT[:, [0, 2], 3]; Q = at(iT_ts, iT_P, tsT + best[0])[:, [0, 2]]; g = np.isfinite(P).all(1) & np.isfinite(Q).all(1); Rr, tt = rigid2(P[g], Q[g]); r = np.linalg.norm(P[g] @ Rr.T + tt - Q[g], axis=1)
print(f"[teach] 11:39 KISS-ICP keyframes vs its INS at {best[0]} ms, rigid: median {np.median(r):.2f} m, p90 {np.percentile(r, 90):.2f}, max {r.max():.2f}")
np.savez('/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad/sameday_pairs_fixed.npz', e=e, al=al, cr=cr, jx=jx, kk=k[ok], offT=best[0], kiss_res=r)

# ---- refine the offsets at 10 ms and draw before / after ----
r1 = own_scan(cR, tsR, iR_ts, iR_P, [0, 1], 6500, 7501, 10); b1 = min(r1, key=lambda r: r[1])
print(f"[refine] 10:54 pose stream vs INS: best {b1[0]} ms, median {b1[1]:.3f} m (within 0.01 m of the best: {[r[0] for r in r1 if r[1] <= b1[1] + 0.01][0]}..{[r[0] for r in r1 if r[1] <= b1[1] + 0.01][-1]} ms)")
rows = []
for offT in range(6500, 7501, 10):
    ok, e = resid(b1[0], offT); rows.append((offT, float(np.median(np.linalg.norm(e[ok], axis=1)))))
b2 = min(rows, key=lambda r: r[1]); print(f"[refine] pairs, 11:39 INS offset with 10:54 at {b1[0]} ms: best {b2[0]} ms, median {b2[1]:.3f} m (within 0.01 m: {[r[0] for r in rows if r[1] <= b2[1] + 0.01][0]}..{[r[0] for r in rows if r[1] <= b2[1] + 0.01][-1]} ms)")
fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True); fig.subplots_adjust(0.09, 0.09, 0.99, 0.88, 0, 0.36)
for a, (offR, offT, ttl) in zip(ax, [(0, 0, 'INS clocks as used (0 s)'), (b1[0], b2[0], f'INS clocks corrected (10:54 +{b1[0] / 1000:.2f} s, 11:39 +{b2[0] / 1000:.2f} s)')]):
    ok, e = resid(offR, offT); n = np.linalg.norm(e[ok], axis=1); tt = (tsR[k[ok]] - tsR[0]) / 1000
    a.plot(tt, n, '.', ms=2.2, color='#2F5D8A'); a.axhline(0.5, color='#6B766E', lw=0.7, ls='--'); a.set_ylim(0, 4); a.set_ylabel('INS distance, m')
    hid = int((n > 4).sum()); a.set_title(f'{ttl}: median {np.median(n):.2f} m, within 0.5 m {100 * (n <= 0.5).mean():.0f} %, beyond 3 m {int((n > 3).sum())} of {len(n)}' + (f' ({hid} above 4 m not drawn)' if hid else ''), fontsize=9, loc='left')
    print(f'[pairs-final] {ttl}: n {len(n)}, median {np.median(n):.3f}, p90 {np.percentile(n, 90):.2f}, within 0.5 m {100 * (n <= 0.5).mean():.1f} %, within 1 m {100 * (n <= 1).mean():.1f} %, > 3 m {int((n > 3).sum())}, max {n.max():.2f}')
ax[1].set_xlabel('time into the 10:54 pass, s'); fig.suptitle('Same-day STPG pairs: GNSS-aided INS distance between each 10:54 keyframe and the 11:39 keyframe it was matched to', fontsize=9.5)
out = '/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/apr1130_vs_1054/ins_check_clock.png'; fig.savefig(out, dpi=160); print('[fig]', out)

# ---- is the 11:39 KISS-ICP teach broken at the lidar gap? piecewise rigid fits vs its INS at +6960 ms ----
P = cT[:, [0, 2], 3]; Q = at(iT_ts, iT_P, tsT + 6960)[:, [0, 2]]; g = np.isfinite(P).all(1) & np.isfinite(Q).all(1); tk = (tsT - tsT[1]) / 1000
def fitres(m):
    R_, t_ = rigid2(P[m], Q[m]); return np.linalg.norm(P[m] @ R_.T + t_ - Q[m], axis=1)
whole = fitres(g); best = None
for s in np.arange(30, tk[g].max() - 30, 2.0):
    a, b = g & (tk < s), g & (tk >= s); r = np.r_[fitres(a), fitres(b)]; m = float(np.median(r))
    if best is None or m < best[1]: best = (float(s), m, float(np.median(fitres(a))), float(np.median(fitres(b))))
print(f"[kiss-break] 11:39 KISS vs INS (+6.96 s): one rigid fit median {np.median(whole):.2f} m; best two-piece split at {best[0]:.0f} s into the keyframes -> median {best[1]:.2f} m (before {best[2]:.2f}, after {best[3]:.2f})")
# speed of each pass from its INS, to convert clock offsets into along-row metres
for nm, ts_, P_ in [('10:54', iR_ts, iR_P[:, :2]), ('11:39', iT_ts, iT_P[:, [0, 2]])]:
    v = np.linalg.norm(np.diff(P_, axis=0), axis=1) / (np.diff(ts_) / 1000); print(f"[speed] {nm} INS median speed {np.median(v[np.isfinite(v)]):.2f} m/s")
