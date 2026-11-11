#!/usr/bin/env python3
"""azalai_rr_odo.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe", Part 3 of RR_design.md (5c6e5a4):
along-row odometry error on the logs. For each lane pass (the lane passes of azalai_rr_prep.py, found again here from the graph's
keyframes) of each named survey: the laser odometry (experimental/laser_dump/lo_poses.npz, plain; KISS-ICP scan poses at 10 Hz)
and the INS positions (experimental/ins_enu_abs.npz) over the pass's time window. With L the distance from the lane's start (the
INS chord), e_odo(L) = | chord of the laser odometry from the start - chord of the INS from the start | (3-D chords). Reported per
pass: e_odo at the lane's end, its maximum over the lane, the largest gap of either stream inside the window, and the INS fix
quality (fixposition fpa/gnsscorr: fix type of the two antennas, 8 = RTK fixed, 7 = RTK float, read from byte 17 / 20 of the CDR
record of prod/monos/monolithics/gnsscorr_raw.npz, header stamp at bytes 4-11). A pass whose INS has no usable fix (no gnsscorr
record in the window, or antenna 1 below RTK float in any of them) or does not cover the window is reported as such and left out.
Clocks: the INS stamp = lidar stamp + offset. The offset is the one stored beside the survey (ins_clock_offset_lidar.json) when
there is one; it is also recomputed here by the stored method (rigid 2-D fit of the scan poses to the INS track over an offset
scan, 100 ms then 10 ms; scans before the largest lidar gap only when that gap exceeds 1 s) and used when none is stored. The lane
window is the pass's first to last keyframe stamp, read on the lidar clock (image and lidar stamps differ by a few tenths of a
second, half a metre of lane at most; the comparison itself uses lidar stamps only). Where patched laser odometry exists
(lo_poses_viopatch*.npz) the same numbers are given for it beside the plain one. No lever arm between the lidar and the INS is
applied (a chord from the start is the same for two points of a rigid body up to the heading change times the lever arm).
nerf_new python3.10:  azalai_rr_odo.py <out rr_odo.json> [--main dec,apr1130] [--extra apr1400,apr16_1230]"""
import argparse, json, struct, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--main', default='dec,apr1130'); ap.add_argument('--extra', default=''); a = ap.parse_args()
X = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental'); E = json.load(open(X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'))['experiences']
def passes(C):
    ok = np.isfinite(C).all((1, 2)); z = C[:, 2, 2]; x = C[:, 0, 3]; runs = []; cur = []
    for k in range(len(C)):
        s = 0 if (not ok[k] or abs(z[k]) <= 0.8) else int(np.sign(z[k]))
        if s != 0 and cur and s == cur[0][1] and k == cur[-1][0] + 1 and abs(x[k] - x[cur[-1][0]]) <= 0.6: cur.append((k, s))
        else:
            if len(cur) >= 30: runs.append(cur)
            cur = [(k, s)] if s != 0 else []
    if len(cur) >= 30: runs.append(cur)
    return [(r[0][0], r[-1][0], r[0][1]) for r in runs]
def rigid2(A, B):
    ca, cb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb)); Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0: Vt[1] *= -1; Rm = Vt.T @ U.T
    return Rm, cb - Rm @ ca
def clock_offset(lts, K, its, IP):
    d = np.diff(lts); g = int(np.argmax(d)); use = (np.arange(len(lts)) <= g) if d[g] > 1000 else np.ones(len(lts), bool)
    def res(off):
        q = lts + off; ok = use & (q >= its[0]) & (q <= its[-1])
        if ok.sum() < 100: return np.inf
        I = np.c_[np.interp(q[ok], its, IP[:, 0]), np.interp(q[ok], its, IP[:, 1])]; Rm, t = rigid2(K[ok], I); return float(np.median(np.linalg.norm(K[ok] @ Rm.T + t - I, axis=1)))
    c = min((res(o), o) for o in range(-15000, 15001, 100))[1]; f = min((res(o), o) for o in range(c - 200, c + 201, 10)); return int(f[1]), float(f[0]), float(d[g]), bool(d[g] > 1000)
OUT = dict(design='RR_design.md 5c6e5a4, Part 3', surveys={}); rows = []
for name in [n for n in (a.main + ',' + a.extra).split(',') if n]:
    R = Path(E[name]['survey']) / 'experimental'; S = dict(root=str(R.parent), main=name in a.main.split(',')); OUT['surveys'][name] = S
    if not (R / 'laser_dump/lo_poses.npz').exists() or not (R / 'ins_enu_abs.npz').exists(): S['skip'] = 'no laser odometry or no INS track on the box'; print(f'[odo] {name}: {S["skip"]}'); continue
    ins = np.load(R / 'ins_enu_abs.npz'); its, IP = ins['ts_ms'].astype(float), ins['xyz'].astype(float); files = {'plain': R / 'laser_dump/lo_poses.npz'}
    for v in ('viopatch', 'viopatch_icp'):
        if (R / f'laser_dump/lo_poses_{v}.npz').exists(): files[v] = R / f'laser_dump/lo_poses_{v}.npz'
    LO = {v: (np.load(f)['ts_ms'].astype(float), np.load(f)['T'][:, :3, 3].astype(float)) for v, f in files.items()}; lts, LP = LO['plain']
    off_c, med_c, gap, split = clock_offset(lts, LP[:, :2], its, IP[:, :2]); st = json.load(open(R / 'ins_clock_offset_lidar.json')) if (R / 'ins_clock_offset_lidar.json').exists() else None
    off = float(st['offset_ms']) if st else float(off_c); S['clock'] = dict(offset_ms_used=off, source='stored ins_clock_offset_lidar.json' if st else 'recomputed here (none stored)', offset_ms_recomputed=off_c, median_m_at_recomputed=med_c, stored=st, largest_lidar_gap_ms=gap)
    gf = R.parent / 'prod/monos/monolithics/gnsscorr_raw.npz'; fx = None
    if gf.exists():
        raw = [np.asarray(r, np.uint8).tobytes() for r in np.load(gf, allow_pickle=True)['raw']]; fx = np.array([[struct.unpack_from('<iI', b, 4)[0] * 1000 + struct.unpack_from('<iI', b, 4)[1] // 1_000_000, b[17], b[20]] for b in raw], float)
    C = np.array(E[name]['c2w'], float); ts = np.array(E[name]['ts_ms'], float); S['passes'] = []
    print(f"[odo] {name}: INS stamp = lidar stamp + {off:.0f} ms ({S['clock']['source']}; recomputed {off_c} ms, median {med_c:.3f} m); largest lidar gap {gap:.0f} ms; laser odometry files {list(files)}", flush=True)
    for i, (k0, k1, sn) in enumerate(passes(C)):
        t0, t1 = ts[k0], ts[k1]; r = dict(survey=name, lane=i + 1, sense=sn, t0_ms=t0, t1_ms=t1, seconds=(t1 - t0) / 1000.0)
        im = (its >= t0 + off) & (its <= t1 + off); r['ins_covers'] = bool(its[0] <= t0 + off and its[-1] >= t1 + off); r['ins_gap_max_ms'] = float(np.diff(its[im]).max()) if im.sum() > 1 else None
        if fx is not None:
            f = fx[(fx[:, 0] >= t0 + off) & (fx[:, 0] <= t1 + off)]; r['fix'] = dict(records=len(f), antenna1_min=int(f[:, 1].min()) if len(f) else None, antenna2_min=int(f[:, 2].min()) if len(f) else None, antenna1_rtk_fixed=int((f[:, 1] == 8).sum()), antenna2_rtk_fixed=int((f[:, 2] == 8).sum()))
        else: r['fix'] = None
        r['usable'] = bool(r['ins_covers'] and r['fix'] is not None and r['fix']['records'] > 0 and r['fix']['antenna1_min'] >= 7)
        if not r['usable']: r['why_not'] = 'INS does not cover the pass' if not r['ins_covers'] else 'no usable INS fix on the pass'
        for v, (lt, lp) in LO.items():
            m = (lt >= t0) & (lt <= t1)
            if m.sum() < 10 or not r['ins_covers']: r[v] = None; continue
            t = lt[m]; p = lp[m]; q = np.c_[[np.interp(t + off, its, IP[:, d]) for d in range(3)]].T; cl = np.linalg.norm(p - p[0], axis=1); ci = np.linalg.norm(q - q[0], axis=1); e = np.abs(cl - ci); j = int(np.argmax(e))
            r[v] = dict(scans=int(m.sum()), L_end_m=float(ci[-1]), lo_end_m=float(cl[-1]), e_end_m=float(e[-1]), e_max_m=float(e[j]), L_at_max_m=float(ci[j]), e_end_pct=float(100 * e[-1] / ci[-1]), lo_gap_max_ms=float(np.diff(t).max()),
                        e_at_quarters_m=[float(e[int(np.argmin(np.abs(ci - f_ * ci[-1])))]) for f_ in (0.25, 0.5, 0.75)])
        S['passes'].append(r); rows.append(r); pl = r['plain']
        print(f"[odo]   lane {i + 1} ({sn:+d}) {r['seconds']:.0f}s fix {r['fix']} usable {r['usable']}: " + ('; '.join(f"{v} L {r[v]['L_end_m']:.1f} m, e_end {r[v]['e_end_m']:.3f} m, e_max {r[v]['e_max_m']:.3f} m at {r[v]['L_at_max_m']:.1f} m, lidar gap {r[v]['lo_gap_max_ms']:.0f} ms" for v in LO if r[v]) or 'no scans'), flush=True)
def stat(R, v, key):
    x = np.array([r[v][key] for r in R]); return dict(n=len(x), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max())) if len(x) else dict(n=0)
def block(names, v='plain'):
    R = [r for r in rows if r['survey'] in names and r['usable'] and r.get(v)]; return dict(passes=len(R), left_out=[(r['survey'], r['lane'], r.get('why_not', 'no scans')) for r in rows if r['survey'] in names and not (r['usable'] and r.get(v))],
                                                                                         e_end_m=stat(R, v, 'e_end_m'), e_max_m=stat(R, v, 'e_max_m'), e_end_pct=stat(R, v, 'e_end_pct'), L_end_m=stat(R, v, 'L_end_m'))
main = [n for n in a.main.split(',') if n]; OUT['summary'] = dict(main_surveys=main, plain=block(main), per_survey={n: block([n]) for n in OUT['surveys'] if 'skip' not in OUT['surveys'][n]})
# the patched laser odometry where it exists, on the same passes (the plain numbers for surveys without a patch)
for v in ('viopatch', 'viopatch_icp'):
    if any(r.get(v) for r in rows if r['survey'] in main):
        R = [dict(r, mix=(r[v] if r.get(v) else r['plain'])) for r in rows if r['survey'] in main and r['usable'] and r['plain']]; OUT['summary'][f'{v}_where_it_exists'] = dict(passes=len(R), e_end_m=stat(R, 'mix', 'e_end_m'), e_max_m=stat(R, 'mix', 'e_max_m'))
json.dump(OUT, open(a.out, 'w'), indent=1); print(json.dumps(OUT['summary'], indent=1))
