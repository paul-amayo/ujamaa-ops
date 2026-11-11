#!/usr/bin/env python3
"""azalai_online_score.py (11 Nov 2026) — RSS Azalai pilot AZ-2 "online along-row station and the counting baseline", step 2: the
simulation and the scores for one (registry, revisit pass) pair. Implements Azalai_RSS/numbers/pilots_20261111/ONLINE_design.md
(5d76dfe). Lane passes, inner rows, lane frames, the bag-to-lane assignment, the standoff d, the spacing S and the truth pairs are
the RR design's, copied from azalai_rr_prep.py. Reads the causal bag lists of azalai_online_prep.py (causal_<pass>_<source>.npz).
Variants (design, "The variants"): M causal matching on laser odometry (primary); a dead reckoning from the row end; b = M on the
graph poses; cp / cm = M started one bag spacing further along / behind; O oracle station (the ceiling); S, Scp, Scm = M with the
re-acquisition search (secondary); with --extra lo_viopatch_icp also Mx, ax on that odometry file; and the counting baseline.
Writes <out>/online_targets_<reg>_<rev>[_direct].jsonl (one line per truth pair, non-targets with their reason),
online_curves_<reg>_<rev>[_direct].json (per lane and keyframe: distance into the lane, station error, offset, matches) and
online_summary_<reg>_<rev>[_direct].json. nerf_new python3.10:
  azalai_online_score.py --reg dec --rev apr1130 --causal <dir> --out <dir> [--pilot h3_arrival_apr1130.json] [--truth direct]
                         [--extra lo_viopatch_icp]"""
import argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from scipy.stats import beta, binomtest
ap = argparse.ArgumentParser(); ap.add_argument('--reg', default='dec'); ap.add_argument('--rev', default='apr1130'); ap.add_argument('--causal', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--pilot', default=''); ap.add_argument('--truth', default='sitewide', choices=['sitewide', 'direct']); ap.add_argument('--extra', default=''); a = ap.parse_args()
KL = Path('/home/paperspace/data/klapmuts'); X = KL / 'dec_2025_ten_rows/experimental'; SK = X / 'sankofa/jirs_20261023/lidar_fixedreader'; STPG = X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'
Z = np.array([0.0, 0.0, 1.0]); AXIS_Z, JUMP, MINLEN, INNER, SEQ_R, PAIR_R = 0.8, 0.6, 30, 1.5, 0.45, 0.5     # constants of the RR design
AX_MIN, STOP_MIN, RESID_MAX = 1.0, 1.0, 0.15                                                              # this design: chord for an axis; stop at least 1 m into the pass; truth readable
SR_WIN, SR_STEP, SR_TOL, SR_MIN, SR_GAIN = 2.0, 0.05, 0.15, 8, 3                                          # variant S (secondary): the four judgements of the design, and the grid step
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); E = json.load(open(STPG))['experiences']; TAG = f'{a.reg}_{a.rev}' + ('_direct' if a.truth == 'direct' else '')
def G_of(name): return np.eye(4) if name == 'dec' else np.load(SK / f'placed_{name}.npz')['G_global'].astype(float)
def resid_of(name): return 0.0 if name == 'dec' else float(np.load(SK / f'placed_{name}.npz')['global_fit_resid_med'])
def tf(T, p): return (T[:3, :3] @ np.asarray(p, float).T).T + T[:3, 3]
# ---------------------------------------------------------------- lane passes, inner rows, lane frames (copied from azalai_rr_prep.py)
def passes(C):
    ok = np.isfinite(C).all((1, 2)); z = C[:, 2, 2]; x = C[:, 0, 3]; runs = []; cur = []
    for k in range(len(C)):
        s = 0 if (not ok[k] or abs(z[k]) <= AXIS_Z) else int(np.sign(z[k]))
        if s != 0 and cur and s == cur[0][1] and k == cur[-1][0] + 1 and abs(x[k] - x[cur[-1][0]]) <= JUMP: cur.append((k, s))
        else:
            if len(cur) >= MINLEN: runs.append(cur)
            cur = [(k, s)] if s != 0 else []
    if len(cur) >= MINLEN: runs.append(cur)
    return [(r[0][0], r[-1][0], r[0][1]) for r in runs]
def lane_frame(Bi, lat, sense):
    left = lat < 0
    if left.sum() < 2 or (~left).sum() < 2: return None
    c = Bi.mean(0); n = np.linalg.svd(Bi - c)[2][2]; n = -n if n[1] > 0 else n
    ea = Z - (Z @ n) * n; ea /= np.linalg.norm(ea); el = np.cross(n, ea); u = (Bi - c) @ el; w = (Bi - c) @ ea
    (m, cl, cr), *_ = np.linalg.lstsq(np.c_[w, left, ~left].astype(float), u, rcond=None)
    ax = ea + m * el; ax /= np.linalg.norm(ax); l = np.cross(n, ax); q0 = c + 0.5 * (cl + cr) * el
    s = (Bi - q0) @ ax; s0 = float(s.min() if sense > 0 else s.max()); F = np.eye(4); F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = l, n, ax, q0 + s0 * ax
    return F, dict(station=s - s0, left=left)
def season(name):
    C = np.array(E[name]['c2w'], float); z = np.load(SK / f'census_{name}.npz'); B = z['bags_xyz'].astype(float); fin = np.isfinite(B).all(1)
    obs = defaultdict(list); frm = defaultdict(set)
    for b, k, n in zip(z['obs_bag'], z['obs_k'], z['obs_n']): obs[int(b)].append((int(k), int(n))); frm[int(k)].add(int(b))
    L = []
    for k0, k1, s in passes(C):
        P = C[k0:k1 + 1, :3, 3]; m = P[:, [0, 2]].mean(0); d = np.linalg.svd(P[:, [0, 2]] - m)[2][0]; d = d * np.sign(d[1]); nrm = np.array([d[1], -d[0]])
        seen = np.array(sorted(b for b in set().union(*[frm[k] for k in range(k0, k1 + 1)]) if fin[b]), int); lat = (B[seen][:, [0, 2]] - m) @ nrm if len(seen) else np.zeros(0)
        inn = np.abs(lat) <= INNER; fr = lane_frame(B[seen[inn]], lat[inn], s) if inn.any() else None
        L.append(dict(k0=k0, k1=k1, sense=s, inner=seen[inn], lat=lat[inn], F=None if fr is None else fr[0], left=None if fr is None else fr[1]['left'], sig=None if fr is None else s * fr[1]['station']))   # sig: the driving station
    return dict(C=C, B=B, fin=fin, obs=obs, L=L)
def assign(S):
    o = {}
    for i, l in enumerate(S['L']):
        for b, la in zip(l['inner'], l['lat']):
            if int(b) not in o or abs(la) < o[int(b)][1]: o[int(b)] = (i, abs(float(la)))
    return {b: v[0] for b, v in o.items()}
def station(F, p): return (np.asarray(p, float) - F[:3, 3]) @ F[:3, 2]
TW, RV = season(a.reg), season(a.rev); Ct, Bt, Cr, Br = TW['C'], TW['B'], RV['C'], RV['B']; NL = len(TW['L'])
assert NL == len(RV['L']) and all(x['sense'] == y['sense'] for x, y in zip(TW['L'], RV['L'])), 'the lane count or the senses differ: stop and report'
lane_t = assign(TW)
gaps = np.concatenate([np.diff(np.sort(l['sig'][l['left'] == sd])) for l in TW['L'] if l['F'] is not None for sd in (True, False)]); SPACING = float(np.median(gaps)); HALF = 0.5 * SPACING
dl = [abs(station(TW['L'][i]['F'], Bt[b]) - station(TW['L'][i]['F'], Ct[max(TW['obs'][b], key=lambda x: x[1])[0], :3, 3])) for b, i in lane_t.items() if TW['L'][i]['F'] is not None and np.isfinite(Ct[max(TW['obs'][b], key=lambda x: x[1])[0]]).all()]
D_ST = float(np.median(dl))
print(f'[online] registry {a.reg}, revisit {a.rev}: {NL} lanes; standoff d = {D_ST:.3f} m (over {len(dl)} registry inner-row bags); spacing S = {SPACING:.3f} m, S/2 = {HALF:.3f} m', flush=True)
# ---------------------------------------------------------------- truth
okt = np.flatnonzero(TW['fin']); okr = np.flatnonzero(RV['fin'])
def mutual(T):   # mutual nearest neighbours within PAIR_R in the ground plane, revisit carried by T into the registry's frame (site-wide: both in December's)
    if a.truth == 'sitewide': St = tf(G_of(a.reg), Bt[okt]); Sr = tf(G_of(a.rev), Br[okr])
    else: St = Bt[okt]; Sr = tf(T, Br[okr])
    dA, jA = cKDTree(Sr[:, [0, 2]]).query(St[:, [0, 2]]); _, jB = cKDTree(St[:, [0, 2]]).query(Sr[:, [0, 2]])
    return {int(okt[i]): int(okr[jA[i]]) for i in range(len(okt)) if dA[i] <= PAIR_R and jB[jA[i]] == i}
G = np.linalg.inv(G_of(a.reg)) @ G_of(a.rev); truth_info = dict(kind=a.truth, sitewide_resid_med_m=dict(registry=resid_of(a.reg), revisit=resid_of(a.rev)))
truth_info['readable_by_rule'] = bool(max(resid_of(a.reg), resid_of(a.rev)) <= RESID_MAX)
if a.truth == 'direct':   # sensitivity of the design: a direct rigid fit between the two passes, started from the composed site-wide fits
    prev = None
    for it in range(1, 21):
        mnn = mutual(G)
        if mnn == prev: break
        prev = mnn; A_ = Br[list(mnn.values())]; B_ = Bt[list(mnn.keys())]; ma, mb = A_.mean(0), B_.mean(0); U, _, Vt = np.linalg.svd((A_ - ma).T @ (B_ - mb)); Rm = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        G = np.eye(4); G[:3, :3] = Rm; G[:3, 3] = mb - Rm @ ma
    res = np.linalg.norm((tf(G, Br[list(mnn.values())]) - Bt[list(mnn.keys())])[:, [0, 2]], axis=1); truth_info.update(direct_iterations=it, direct_resid_med_m=float(np.median(res)), direct_readable=bool(np.median(res) <= RESID_MAX))
else: mnn = mutual(G)
if a.pilot:
    pairs = [(r['dec'], r['apr']) for r in json.load(open(a.pilot))['targets']]; assert all(mnn.get(t) == p for t, p in pairs), "the pilot's pairs are not mutual nearest neighbours here"
else: pairs = sorted(mnn.items())
truth_info.update(n_pairs=len(pairs), mutual_nn=len(mnn), source='pilot record' if a.pilot else 'mutual nearest neighbours'); print(f'[online] truth: {truth_info}', flush=True)
# ---------------------------------------------------------------- the causal runs
def load(src):
    f = Path(a.causal) / f'causal_{a.rev}_{src}.npz'
    if not f.exists(): return None
    z = np.load(f, allow_pickle=True); assert (z['lanes'] == np.array([(l['k0'], l['k1'], l['sense']) for l in RV['L']])).all(); return {k: z[k] for k in z.files}
def geom(cz, i):
    """per keyframe of lane i: the causal axis and, in it, the robot's odometric station and the inner-row bags seen so far (station, side)"""
    l = RV['L'][i]; k0, k1, s = l['k0'], l['k1'], l['sense']
    if cz is None or not cz['lane_ok'][i]: return None
    W = cz['W']; P = W[k0:k1 + 1, :3, 3]; Yd = W[k0:k1 + 1, :3, 1]; idx = np.flatnonzero(cz['step_lane'] == i); assert len(idx) == k1 - k0 + 1 and (cz['step_k'][idx] == np.arange(k0, k1 + 1)).all(); out = []
    for j, st in enumerate(idx):
        if np.linalg.norm(P[j] - P[0]) < AX_MIN: out.append(None); continue
        Q = P[:j + 1]; cm = Q.mean(0); ax = np.linalg.svd(Q - cm)[2][0]; ax = ax * np.sign(ax @ (P[j] - P[0])); up = -Yd[:j + 1].mean(0); up = up - (up @ ax) * ax; up /= np.linalg.norm(up); rt = np.cross(ax, up)
        Xb = cz['bags_xyz'][cz['step_start'][st]:cz['step_start'][st] + cz['step_n'][st]]; o = (Xb - P[0]) @ ax; lat = (Xb - cm) @ rt; inn = np.abs(lat) <= INNER
        out.append(dict(o_robot=float((P[j] - P[0]) @ ax), o=o[inn], left=(lat[inn] * s < 0), n_seen=len(Xb)))
    return out
def seq(delta, o, left, ts, tl):
    """the sequence rule of the RR design, from delta: per side mutual nearest neighbours within SEQ_R, move by the median difference, until under 1 mm, at most 20 times"""
    nm = 0
    for it in range(20):
        df = []
        for sd in (True, False):
            x, y = ts[tl == sd], o[left == sd] + delta
            if not len(x) or not len(y): continue
            D = np.abs(x[:, None] - y[None]); i_ = D.argmin(1); j_ = D.argmin(0)
            df += [x[p] - y[i_[p]] for p in range(len(x)) if j_[i_[p]] == p and D[p, i_[p]] <= SEQ_R]
        nm = len(df)
        if not df: break
        step = float(np.median(df)); delta += step
        if abs(step) < 1e-3: break
    return delta, nm
SHIFTS = np.round(np.arange(-SR_WIN, SR_WIN + 1e-9, SR_STEP), 6)
def search(delta, o, left, ts, tl):
    """variant S: count the bags seen with a same-side registry bag within SR_TOL at each shift; jump if the best count is >= SR_MIN and beats zero shift by >= SR_GAIN"""
    cnt = np.zeros(len(SHIFTS), int)
    for sd in (True, False):
        x, y = ts[tl == sd], o[left == sd] + delta
        if len(x) and len(y): cnt += (np.abs(y[None, :, None] + SHIFTS[:, None, None] - x[None, None, :]).min(2) <= SR_TOL).sum(1)
    c0 = cnt[np.argmin(np.abs(SHIFTS))]; best = cnt.max()
    if best >= SR_MIN and best >= c0 + SR_GAIN: cand = SHIFTS[cnt == best]; return delta + float(cand[np.argmin(np.abs(cand))]), True
    return delta, False
def run(GS, ts, tl, mode, shift0=0.0, srch=False):
    """-> belief before the update (bm), after (bp), offset, matches, jumps per keyframe of the lane (nan / 0 before a belief exists)"""
    n = len(GS); bm = np.full(n, np.nan); bp = np.full(n, np.nan); dlt = np.full(n, np.nan); nm = np.zeros(n, int); jm = np.zeros(n, int); delta = None
    for j, g in enumerate(GS):
        if g is None: continue
        prev = delta
        if mode == 'dr':
            if len(g['o']): delta = -float(g['o'].min())
        else:
            if delta is None and len(g['o']): delta = -float(g['o'].min()) + shift0
            if delta is not None:
                if srch: delta, jumped = search(delta, g['o'], g['left'], ts, tl); jm[j] = int(jumped)
                delta, nm[j] = seq(delta, g['o'], g['left'], ts, tl)
        if prev is not None: bm[j] = g['o_robot'] + prev
        if delta is not None: bp[j] = g['o_robot'] + delta; dlt[j] = delta
    return dict(bm=bm, bp=bp, delta=dlt, nmatch=nm, jump=jm)
def stop_at(bm, bp, target):
    """first point where the belief reaches the stop station -> (keyframe index j in the lane, fraction f of the interval j-1..j, how) or None"""
    fin = np.flatnonzero(np.isfinite(bp))
    if not len(fin): return None
    j0 = int(fin[0])
    if bp[j0] >= target: return j0, 1.0, 'already past at the first belief'
    for j in range(j0 + 1, len(bp)):
        if np.isfinite(bm[j]) and bm[j] >= target: return j, float((target - bp[j - 1]) / (bm[j] - bp[j - 1])), 'between keyframes'
        if np.isfinite(bp[j]) and bp[j] >= target: return j, 1.0, 'carried past by an update'
    return None
CZ = {s: load(s) for s in ['lo', 'graph'] + ([a.extra] if a.extra else [])}; assert CZ['graph'] is not None, 'graph causal file missing'
VARS = [('M', 'lo', 'seq', 0.0, False), ('a', 'lo', 'dr', 0.0, False), ('b', 'graph', 'seq', 0.0, False), ('cp', 'lo', 'seq', +1.0, False), ('cm', 'lo', 'seq', -1.0, False),
        ('S', 'lo', 'seq', 0.0, True), ('Scp', 'lo', 'seq', +1.0, True), ('Scm', 'lo', 'seq', -1.0, True)] + ([('Mx', a.extra, 'seq', 0.0, False), ('ax', a.extra, 'dr', 0.0, False)] if a.extra else [])
VN = [v[0] for v in VARS] + ['O']; lanes = []; SIM = {}
for i in range(NL):
    t, r = TW['L'][i], RV['L'][i]; k0, k1, s = r['k0'], r['k1'], r['sense']; e = dict(lane=i + 1, sense=s, k0=k0, k1=k1)
    if t['F'] is None or r['F'] is None: e['skip'] = 'an inner row is missing on one side: no lane frame'; lanes.append(e); SIM[i] = None; continue
    Pg = Cr[k0:k1 + 1, :3, 3]; sig_true = s * station(t['F'], tf(G, Pg)); vs = s * station(r['F'], Pg); Ldist = sig_true - sig_true[0]
    sim = dict(sig_true=sig_true, vs=vs, L=Ldist, Pg=Pg, var={}); geoms = {src: geom(cz, i) for src, cz in CZ.items()}
    for name, src, mode, sh, srch in VARS:
        if geoms.get(src) is None: sim['var'][name] = None; continue
        sim['var'][name] = run(geoms[src], t['sig'], t['left'], mode, sh * SPACING, srch)
    sim['var']['O'] = dict(bm=sig_true.copy(), bp=sig_true.copy(), delta=np.full(len(Pg), np.nan), nmatch=np.zeros(len(Pg), int), jump=np.zeros(len(Pg), int))
    e.update(n_keyframes=len(Pg), length_m=float(Ldist[-1]), registry_inner=dict(left=int(t['left'].sum()), right=int((~t['left']).sum())), revisit_inner=dict(left=int(r['left'].sum()), right=int((~r['left']).sum())), var={})
    for name in VN:
        v = sim['var'][name]
        if v is None: e['var'][name] = None; continue
        err = v['bp'] - sig_true; ok = np.isfinite(err); v['err'] = err
        if not ok.any(): e['var'][name] = dict(no_belief=True); continue
        j0 = int(np.flatnonzero(ok)[0]); ae = np.abs(err[ok]); bad = np.flatnonzero(ok & (np.abs(err) >= HALF))
        rec = 'never' if (len(bad) and bad[-1] == len(err) - 1) else (float(Ldist[j0]) if not len(bad) else float(Ldist[bad[-1] + 1]))   # distance after which |e| stays below S/2 to the lane's end
        e['var'][name] = dict(first_belief_at_m=float(Ldist[j0]), err_first_m=float(err[j0]), datum_failure=bool(abs(err[j0]) > HALF), err_end_m=float(err[ok][-1]), abs_med_m=float(np.median(ae)), abs_p90_m=float(np.percentile(ae, 90)), abs_max_m=float(ae.max()),
                              share_over_half=float((ae > HALF).mean()), n_belief=int(ok.sum()), stays_below_half_after_m=rec, matches_last=int(v['nmatch'][-1]), jumps=int(v['jump'].sum()))
    lanes.append(e); SIM[i] = sim
    print(f"[online] lane {i + 1} ({s:+d}) {len(Pg)} kf, {Ldist[-1]:.1f} m: " + '; '.join(f"{n} first {e['var'][n]['err_first_m']:+.2f} end {e['var'][n]['err_end_m']:+.2f} max {e['var'][n]['abs_max_m']:.2f}" for n in VN if e['var'][n] and 'err_first_m' in e['var'][n]), flush=True)
# ---------------------------------------------------------------- targets, outcomes
def outcome(arr, p, order):   # order: the revisit's inner-row bags of the lane and side, sorted by driving station
    if arr is None: return 'other'
    if arr == p: return 'right'
    return 'neighbour' if abs(order.index(arr) - order.index(p)) == 1 else 'other'
recs = []
for t, p in pairs:
    i = lane_t.get(t); e = dict(registry=a.reg, revisit=a.rev, registry_bag=int(t), partner=int(p), lane=None if i is None else i + 1)
    if i is None: e['skip'] = 'target does not stand in an inner row of a lane'; recs.append(e); continue
    Lt, Lr = TW['L'][i], RV['L'][i]
    if Lt['F'] is None or Lr['F'] is None: e['skip'] = 'the lane has no lane frame in one of the passes'; recs.append(e); continue
    jp = np.flatnonzero(Lr['inner'] == p)
    if not len(jp): e['skip'] = "the partner does not stand in the revisit's inner rows of the same lane"; recs.append(e); continue
    jt = int(np.flatnonzero(Lt['inner'] == t)[0]); jp = int(jp[0]); side = bool(Lt['left'][jt])
    if bool(Lr['left'][jp]) != side: e['skip'] = 'the target and its partner are on different sides'; recs.append(e); continue
    s = Lt['sense']; sg = float(Lt['sig'][jt]); n = 1 + int(((Lt['left'] == side) & (Lt['sig'] < sg)).sum()); same = np.flatnonzero(Lr['left'] == side); order = [int(Lr['inner'][q]) for q in same[np.argsort(Lr['sig'][same])]]; vs_of = {int(Lr['inner'][q]): float(Lr['sig'][q]) for q in same}
    rank_p = order.index(p) + 1; nth = order[n - 1] if n <= len(order) else None
    e.update(sense=s, side='left' if side else 'right', station_m=sg, stop_station_m=sg - D_ST, n=n, partner_station_rev_m=vs_of[p], partner_rank_rev=rank_p,
             counting=dict(arrival=nth, outcome=outcome(nth, p, order), count_offset=rank_p - n, no_nth_bag=nth is None))
    sim = SIM[i]; ideal = vs_of[p] - D_ST; e['ideal_stop_rev_m'] = ideal; e['stop_evaluable'] = bool(sim['vs'][0] + STOP_MIN <= ideal <= sim['vs'][-1])
    if not e['stop_evaluable']: e['stop_skip'] = 'stop outside the logged lane pass (' + ('before 1 m into it' if ideal < sim['vs'][0] + STOP_MIN else 'after its last keyframe') + ')'
    e['var'] = {}
    for name in VN:
        v = sim['var'][name]
        if v is None: e['var'][name] = dict(skip='no laser odometry for the lane'); continue
        st = stop_at(v['bm'], v['bp'], sg - D_ST)
        if st is None: e['var'][name] = dict(outcome='other', no_stop=True); continue
        j, f, how = st; pos = sim['Pg'][j] if f >= 1.0 or j == 0 else (1 - f) * sim['Pg'][j - 1] + f * sim['Pg'][j]; vstop = float(Lr['sense'] * station(Lr['F'], pos)); aim = vstop + D_ST
        arr = min(order, key=lambda b: abs(vs_of[b] - aim)); sig_stop = float(s * station(Lt['F'], tf(G, pos)))
        e['var'][name] = dict(outcome=outcome(arr, p, order), no_stop=False, arrival=arr, stop_k=int(Lr['k0'] + j), stop_frac=f, how=how, stop_rev_station_m=vstop, aim_rev_station_m=aim, miss_m=aim - vs_of[p],
                              true_station_at_stop_m=sig_stop, station_error_at_stop_m=(sg - D_ST) - sig_stop, distance_into_lane_m=float(sig_stop - sim['sig_true'][0]))
    recs.append(e)
with open(OUT / f'online_targets_{TAG}.jsonl', 'w') as f:
    for e in recs: f.write(json.dumps(e) + '\n')
# ---------------------------------------------------------------- summary
def cp(k, n): return [float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0] if n else [None, None]
def rate(rows, get):
    c = Counter(get(e) for e in rows); n = len(rows); k = c.get('right', 0); return dict(n=n, right=k, neighbour=c.get('neighbour', 0), other=c.get('other', 0), rate=(k / n if n else None), ci95=cp(k, n))
def sign(rows, ga, gb):
    x = sum(ga(e) == 'right' and gb(e) != 'right' for e in rows); y = sum(ga(e) != 'right' and gb(e) == 'right' for e in rows); return dict(n=len(rows), first_only=x, second_only=y, p=float(binomtest(x, x + y, 0.5).pvalue) if x + y else 1.0)
T = [e for e in recs if 'skip' not in e]; SE = [e for e in T if e['stop_evaluable']]
def has(e, n): return 'outcome' in e['var'][n]
summ = dict(registry=a.reg, revisit=a.rev, design='ONLINE_design.md 5d76dfe', truth=truth_info, standoff_d_m=D_ST, standoff_n=len(dl), spacing_m=SPACING, half_spacing_m=HALF, n_lanes=NL,
            lo_file={k: (str(v['source_file']) if v is not None else None) for k, v in CZ.items()}, lanes_with_lo=[i + 1 for i in range(NL) if CZ['lo'] is not None and CZ['lo']['lane_ok'][i]],
            n_pairs=len(pairs), targets=len(T), not_target=dict(Counter(e['skip'] for e in recs if 'skip' in e)), stop_evaluable=len(SE), not_stop_evaluable=dict(Counter(e['stop_skip'] for e in T if not e['stop_evaluable'])),
            counting=dict(all_targets=rate(T, lambda e: e['counting']['outcome']), stop_evaluable=rate(SE, lambda e: e['counting']['outcome']), no_nth_bag=sum(e['counting']['no_nth_bag'] for e in T),
                          by_n={f'{lo}-{hi if hi < 999 else "up"}': dict(**rate([e for e in T if lo <= e['n'] <= hi], lambda e: e['counting']['outcome']), abs_offset_med=(float(np.median([abs(e['counting']['count_offset']) for e in T if lo <= e['n'] <= hi])) if any(lo <= e['n'] <= hi for e in T) else None))
                                for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 999))},
                          by_lane={str(i + 1): rate([e for e in T if e['lane'] == i + 1], lambda e: e['counting']['outcome']) for i in range(NL)}), variants={}, paired={}, lanes=lanes)
for name in VN:
    rows = [e for e in SE if has(e, name)]; g = lambda e, n=name: e['var'][n]['outcome']; ms = np.array([abs(e['var'][name]['miss_m']) for e in rows if not e['var'][name]['no_stop']])
    errs = np.concatenate([SIM[i]['var'][name]['err'] for i in range(NL) if SIM[i] is not None and SIM[i]['var'][name] is not None] or [np.zeros(0)]); Ls = np.concatenate([SIM[i]['L'] for i in range(NL) if SIM[i] is not None and SIM[i]['var'][name] is not None] or [np.zeros(0)]); ok = np.isfinite(errs)
    summ['variants'][name] = dict(**rate(rows, g), no_stop=sum(e['var'][name]['no_stop'] for e in rows), skipped_no_lo=sum(not has(e, name) for e in SE), how=dict(Counter(e['var'][name].get('how', 'no stop') for e in rows)),
                                  abs_miss_med_m=float(np.median(ms)) if len(ms) else None, abs_miss_p90_m=float(np.percentile(ms, 90)) if len(ms) else None,
                                  by_lane={str(i + 1): rate([e for e in rows if e['lane'] == i + 1], g) for i in range(NL)},
                                  station_error=dict(n_keyframes=int(ok.sum()), abs_med_m=float(np.median(np.abs(errs[ok]))) if ok.any() else None, abs_p90_m=float(np.percentile(np.abs(errs[ok]), 90)) if ok.any() else None, abs_max_m=float(np.abs(errs[ok]).max()) if ok.any() else None,
                                                     share_over_half=float((np.abs(errs[ok]) > HALF).mean()) if ok.any() else None, datum_failures=[l['lane'] for l in lanes if l.get('var', {}).get(name) and l['var'][name].get('datum_failure')],
                                                     bands_5m={f'{b}-{b + 5}': (dict(n=int(m.sum()), abs_med_m=float(np.median(np.abs(errs[m]))), abs_p90_m=float(np.percentile(np.abs(errs[m]), 90)), share_over_half=float((np.abs(errs[m]) > HALF).mean())) if m.any() else None)
                                                               for b in range(0, 50, 5) for m in [ok & (Ls >= b) & (Ls < b + 5)]}))
for x, y in (('M', 'counting'), ('M', 'a'), ('M', 'b'), ('M', 'O'), ('b', 'O'), ('a', 'counting'), ('S', 'M'), ('Mx', 'M'), ('Mx', 'b')):
    gx = (lambda e, n=x: e['counting']['outcome']) if x == 'counting' else (lambda e, n=x: e['var'][n]['outcome']); gy = (lambda e, n=y: e['counting']['outcome']) if y == 'counting' else (lambda e, n=y: e['var'][n]['outcome'])
    if (x != 'counting' and x not in VN) or (y != 'counting' and y not in VN): continue
    rows = [e for e in SE if (x == 'counting' or has(e, x)) and (y == 'counting' or has(e, y))]; summ['paired'][f'{x}_vs_{y}'] = dict(**sign(rows, gx, gy), first=rate(rows, gx), second=rate(rows, gy))
json.dump(summ, open(OUT / f'online_summary_{TAG}.json', 'w'), indent=1)
json.dump(dict(registry=a.reg, revisit=a.rev, half_spacing_m=HALF, lanes=[dict(lane=i + 1, L=np.round(SIM[i]['L'], 4).tolist(), sig_true=np.round(SIM[i]['sig_true'], 4).tolist(),
          var={n: (None if SIM[i]['var'][n] is None else dict(err=[None if not np.isfinite(x) else round(float(x), 4) for x in SIM[i]['var'][n]['err']], delta=[None if not np.isfinite(x) else round(float(x), 4) for x in SIM[i]['var'][n]['delta']], nmatch=SIM[i]['var'][n]['nmatch'].tolist(), jump=SIM[i]['var'][n]['jump'].tolist())) for n in VN}) for i in range(NL) if SIM[i] is not None]),
          open(OUT / f'online_curves_{TAG}.json', 'w'))
print(json.dumps({k: v for k, v in summ.items() if k not in ('lanes', 'variants', 'counting')}, indent=1)); print('counting', json.dumps(summ['counting']['all_targets']), json.dumps(summ['counting']['stop_evaluable']))
for n in VN: v = summ['variants'][n]; print(f"{n:4s} right {v['right']}/{v['n']} neighbour {v['neighbour']} other {v['other']} (no stop {v['no_stop']}) rate {v['rate'] if v['rate'] is None else round(v['rate'], 3)} ci {v['ci95']}; station |e| med {v['station_error']['abs_med_m']} p90 {v['station_error']['abs_p90_m']} max {v['station_error']['abs_max_m']} over S/2 {v['station_error']['share_over_half']}")
