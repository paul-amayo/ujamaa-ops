#!/usr/bin/env python3
"""twin_ledger_score.py (11 Nov 2026, A100 session) - JIRS test JS-1, step 3 of 5: the twin-read record of every plant address
and the tests T-id1 to T-id4 with the two repeats, exactly as the frozen design says (DESIGN.md, JIRS 37e263f; copy committed
4ef2ab0) and the implementation addendum (ADDENDUM_A100_implementation.md, baf5123, committed before any score). CPU only.
Inputs: prep.json (twin_ledger_prep.py), the point files of twin_ledger_render.py, the paper's lidar ledger (census, placed,
heights of the fixed-reader rebuild). Outputs: <records>/plants_{dec,apr}.jsonl, pairs.jsonl, split_half.jsonl,
cross_cell.jsonl, panels_selected.json; <results>/summary.json.
Addendum A100-2 (7023dad): the twin lane constant is the design's rule without the opacity cut (twin_ledger_lane.py files);
the render's opacity-cut constant is carried as the deviation line 'height_lanecut'.
  python3 twin_ledger_score.py <prep.json> <points dir> <records dir> <results dir> <lane dir>"""
import sys, json, time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

t0 = time.time()
PREP, PTS, REC, RES, LANED = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
REC.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
P = json.load(open(PREP)); L = Path(P['census_dir'])
# every constant is the paper's or the design's (DESIGN.md; ADDENDUM items in brackets)
OBS_MIN = 10      # a keyframe observes an address with >= 10 points (bag_census_pass.py)
POS_R = 0.35      # census assignment radius: the second median is over the points within 0.35 m of the first
TOP_R = 0.30      # plant-top column radius (bag_heights_proj.py)
TOP_MIN = 20      # points in the column for a keyframe to give a top (bag_heights_proj.py)
TOP_K = 12        # up to 12 observing keyframes with the most points (bag_heights_proj.py)
PAIR_R = 0.5      # association radius, mutual nearest neighbours on the ground plane (ledger_matrix.py)
THR = 0.082       # growth threshold, largest same-day p95 of |height difference| (heights_repeatability.json)
T1_SHARE = 0.95   # T-id1 verdict share (design: the paper's own percentile, a judgement)
SEASONS = {'dec': 'dec', 'apr': 'apr1054'}
CALLS = ['grown', 'stable', 'shrunk']


def log(*a): print(f'[score {time.time() - t0:6.0f}s]', *a, flush=True)


def q(x, p): return float(np.percentile(np.asarray(x, float), p)) if len(x) else None


def stats_abs(d):
    d = np.asarray(d, float)
    if not len(d): return dict(n=0)
    return dict(n=int(len(d)), bias_median=float(np.median(d)), abs_median=float(np.median(np.abs(d))), abs_p95=q(np.abs(d), 95), abs_max=float(np.abs(d).max()))


def stats_pos(d):
    d = np.asarray(d, float)
    if not len(d): return dict(n=0)
    return dict(n=int(len(d)), median=float(np.median(d)), p90=q(d, 90), p95=q(d, 95), max=float(d.max()))


def call_of(c): return 'grown' if c > THR else ('shrunk' if c < -THR else 'stable')


def confusion(rows, cols):
    """3 x 3 confusion, rows = first argument's calls, columns = second's, order grown, stable, shrunk"""
    M = np.zeros((3, 3), int)
    for r, c in zip(rows, cols): M[CALLS.index(r), CALLS.index(c)] += 1
    n = int(M.sum()); return dict(n=n, matrix=M.tolist(), agree=int(np.trace(M)), agreement=(float(np.trace(M) / n) if n else None))


def mutual_nn(A, B, thr=PAIR_R):
    """make_figures.py's mutual_nn: mutual nearest neighbours within thr in 2-D, non-finite rows left out, original indices"""
    fa, fb = np.flatnonzero(np.isfinite(A).all(1)), np.flatnonzero(np.isfinite(B).all(1)); Af, Bf = A[fa], B[fb]
    ta, tb = cKDTree(Af), cKDTree(Bf); dB, iB = ta.query(Bf); dA, iA = tb.query(Af)
    pairs = [(fa[a], fb[iA[a]], dA[a]) for a in range(len(Af)) if dA[a] <= thr and iB[iA[a]] == a]
    return np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int), np.array([p[2] for p in pairs], float)


def record(xyz, k, up, camh, lanes):
    """The twin-read record of one address from its points (any subset): DESIGN 'The twin-read record of a plant'; ADDENDUM 2-4."""
    r = dict(n_points=int(len(k)))
    if len(k) == 0: r['reason'] = 'no point reads the address'; return r
    ku, inv, cnt = np.unique(k, return_inverse=True, return_counts=True); obs = cnt >= OBS_MIN
    r['n_kf'] = int(len(ku)); r['n_obs_kf'] = int(obs.sum())
    if not obs.any(): r['reason'] = 'no observing keyframe (>= 10 points)'; return r
    m = obs[inv]; Q = xyz[m]; r['n_obs_points'] = int(m.sum())
    m1 = np.median(Q, 0); near = np.hypot(Q[:, 0] - m1[0], Q[:, 2] - m1[2]) <= POS_R
    if not near.any(): r['reason'] = 'no point within 0.35 m of the first median'; return r
    pos = np.median(Q[near], 0); r['pos'] = [float(v) for v in pos]; r['n_pos_points'] = int(near.sum())
    cand = sorted(((int(c), int(kk)) for kk, c, o in zip(ku, cnt, obs) if o), reverse=True)[:TOP_K]
    tops, chs, used = [], [], []
    for c, kk in cand:
        Qk = xyz[k == kk]; col = np.hypot(Qk[:, 0] - pos[0], Qk[:, 2] - pos[2]) <= TOP_R
        if col.sum() >= TOP_MIN: tops.append(float(np.percentile(Qk[col] @ up, 95))); chs.append(camh[kk]); used.append(kk)
    r['top_kf'] = used; r['top_kf_values'] = tops; r['n_top_kf'] = len(used)
    if not used: r['height_reason'] = 'no keyframe with >= 20 points in the 0.30 m column'; return r
    top = float(np.median(tops)); ch = float(np.median(chs))
    r.update(top=top, cam_h=ch, **{name: top - (ch - lc) for name, lc in lanes.items()}); return r


# ------------------------------------------------------------------ the lidar ledger (the paper's records)
cen = {s: np.load(L / f'census_{p}.npz', allow_pickle=True) for s, p in SEASONS.items()}
BXYZ = {s: cen[s]['bags_xyz'].astype(float) for s in SEASONS}
HZ = {s: np.load(L / f'heights_{p}.npz', allow_pickle=True) for s, p in SEASONS.items()}
for s in SEASONS:
    assert np.allclose(np.array(P[s]['up']), HZ[s]['up']), f'{s}: up differs from heights npz'
    assert abs(P[s]['mount_lane_lidar'] - float(HZ[s]['mount_lane'])) < 1e-9
PL54 = np.load(L / 'placed_apr1054.npz', allow_pickle=True); G = np.array(P['G_global'], float)
assert np.allclose(G, PL54['G_global']), 'G_global in prep differs from placed_apr1054.npz'
# lidar pairs (ADDENDUM 5): census_dec bags_xyz x/z against placed_apr1054 global x/z
LD, LA, LDIST = mutual_nn(BXYZ['dec'][:, [0, 2]], PL54['global'][:, [0, 2]])
assert len(LD) == 680, f'lidar pairs {len(LD)} != the paper 680'
hd_dec, hd_apr = HZ['dec']['height_cam'].astype(float), HZ['apr']['height_cam'].astype(float)
L1chg = hd_apr[LA] - hd_dec[LD]; okL1 = np.isfinite(L1chg)
assert okL1.sum() == 677 and (L1chg[okL1] > THR).sum() == 192 and (L1chg[okL1] < -THR).sum() == 123, 'L1 does not reproduce heights_repeatability.json dec|apr1054'
L1 = {int(d): dict(a=int(a), dist=float(dd), change=(float(c) if np.isfinite(c) else None), call=(call_of(c) if np.isfinite(c) else None)) for d, a, dd, c in zip(LD, LA, LDIST, L1chg)}
lidar_pair_set = {(int(d), int(a)) for d, a in zip(LD, LA)}
# L2 = make_figures.py lines 96-115 verbatim (December positions census_dec bags_xz)
DXZ = cen['dec']['bags_xz'].astype(float)
placed = {p: np.load(L / f'placed_{p}.npz', allow_pickle=True) for p in ['apr1130', 'apr1400', 'apr16_1230']}
heights = {p: np.load(L / f'heights_{p}.npz', allow_pickle=True) for p in ['apr1130', 'apr1400', 'apr16_1230']}; heights['dec'] = HZ['dec']
thr = float(json.load(open(L / 'jirs_summary.json'))['heights']['threshold_m']); assert thr == THR


def dh(p, q_):
    A = placed[p]['global'][:, [0, 2]] if p != 'dec' else DXZ; B = placed[q_]['global'][:, [0, 2]] if q_ != 'dec' else DXZ
    ia, ib, _ = mutual_nn(A, B); hp, hq = heights[p]['height_cam'], heights[q_]['height_cam']; d = hq[ib] - hp[ia]; ok = np.isfinite(d); return d[ok], ia[ok]


votes = np.zeros((len(DXZ), 3), int)
for p in ['apr1130', 'apr1400', 'apr16_1230']:
    d, ia = dh('dec', p); votes[ia[d > thr], 0] += 1; votes[ia[np.abs(d) <= thr], 1] += 1; votes[ia[d < -thr], 2] += 1
seen = votes.sum(1) > 0; call = votes.argmax(1)
L2 = {int(b): CALLS[int(call[b])] for b in np.flatnonzero(seen)}
L2c = Counter(L2.values()); assert len(L2) == 728 and L2c['grown'] == 227 and L2c['stable'] == 366 and L2c['shrunk'] == 135, f'L2 does not reproduce the paper: {len(L2)} {L2c}'
log(f'lidar ledger reproduced: 680 pairs Dec|apr1054; L1 n 677 grown 192 shrunk 123; L2 728 called {dict(L2c)}')

# ------------------------------------------------------------------ twin-read records, per season and method
R = {}; LANE = {}; CELLS = {}; RAW = {}
for s in SEASONS:
    S = P[s]; up = np.array(S['up'], float); camh = {int(k): float(np.array(v, float) @ up) for k, v in S['cam_centres_graph'].items()}
    cells = sorted(S['cells']); CELLS[s] = cells; Z = {c: np.load(PTS / f'{s}_{c}.npz') for c in cells}
    LZ = {c: np.load(LANED / f'{s}_{c}.npz') for c in cells}
    for c in cells:   # the lane script reads the same depth as the render: its opacity-cut values must equal the render's lane_h
        assert np.array_equal(LZ[c]['k_cut'], Z[c]['lane_k']) and np.allclose(LZ[c]['h_cut'], Z[c]['lane_h'], atol=1e-5, rtol=0), f'{s} {c}: lane script does not reproduce the render lane_h'
    lane_all = np.concatenate([LZ[c]['h_nocut'].astype(float) for c in cells]); lane_cut = np.concatenate([Z[c]['lane_h'].astype(float) for c in cells])
    LANE[s] = dict(twin=float(np.median(lane_all)), n_frames=int(len(lane_all)), iqr=float(np.subtract(*np.percentile(lane_all, [75, 25]))), per_cell={c: (float(np.median(LZ[c]['h_nocut'])) if len(LZ[c]['h_nocut']) else None) for c in cells},
                   opacity_cut=float(np.median(lane_cut)), n_frames_opacity_cut=int(len(lane_cut)), alpha_band_median=float(np.nanmedian(np.concatenate([LZ[c]['alpha_band_median'] for c in cells]))), lidar=float(S['mount_lane_lidar']))
    lanes = dict(height=LANE[s]['twin'], height_lidarlane=LANE[s]['lidar'], height_lanecut=LANE[s]['opacity_cut']); vocab = [int(v) for v in S['vocab']]; R[s] = {}
    log(f"{s}: lane constant twin {LANE[s]['twin']:.4f} m over {len(lane_all)} frames (design rule); opacity-cut {LANE[s]['opacity_cut']:.4f} m; lidar {LANE[s]['lidar']:.4f}; vocabulary {len(vocab)}")
    for m in ('b', 'a'):
        X = np.concatenate([Z[c][f'{m}_xyz'] for c in cells]).astype(np.float64); A = np.concatenate([Z[c][f'{m}_addr'] for c in cells]).astype(np.int64)
        K = np.concatenate([Z[c][f'{m}_k'] for c in cells]).astype(np.int64); C = np.concatenate([np.full(len(Z[c][f'{m}_addr']), i, np.int64) for i, c in enumerate(cells)])
        o = np.lexsort((K, A)); X, A, K, C = X[o], A[o], K[o], C[o]; ua, st, ct = np.unique(A, return_index=True, return_counts=True); sl = {int(a): slice(int(i), int(i + n)) for a, i, n in zip(ua, st, ct)}
        stray = sorted(set(sl) - set(vocab)); assert not stray, f'{s} {m}: addresses outside the vocabulary {stray[:5]}'
        recs = {}
        for a in vocab:
            if a not in sl: recs[a] = dict(n_points=0, reason='no point reads the address', cells={}); continue
            x, k, c = X[sl[a]], K[sl[a]], C[sl[a]]; r = record(x, k, up, camh, lanes); r['cells'] = {cells[i]: int(n) for i, n in zip(*np.unique(c, return_counts=True))}
            if m == 'b':   # repeats (method b only, ADDENDUM 11)
                ku, cnt = np.unique(k, return_counts=True); ok = ku[cnt >= OBS_MIN]
                if len(ok) >= 2:
                    h1, h2 = np.isin(k, ok[0::2]), np.isin(k, ok[1::2]); r['split'] = [record(x[h1], k[h1], up, camh, lanes), record(x[h2], k[h2], up, camh, lanes)]
                r['per_cell'] = {cells[i]: record(x[c == i], k[c == i], up, camh, lanes) for i in np.unique(c)}
            recs[a] = r
        R[s][m] = recs
        if m == 'b': RAW[s] = (A, K, C)   # for the panel cell choice
        nread = sum('pos' in r for r in recs.values()); nh = sum('height' in r for r in recs.values())
        log(f'{s} method {m}: {len(X):,} points, {len(sl)} addresses with points; position for {nread} of {len(vocab)}, height for {nh}; reasons {dict(Counter(r.get("reason", "read") for r in recs.values()))}')
        del X

# ------------------------------------------------------------------ naming
NAME = {}
for s in SEASONS:
    nm = P[s]['naming']; NAME[s] = {int(b): dict(identity=int(v['identity']), in_vocab=bool(v['in_vocab']), vote_agrees=v['vote_agrees'], anchor_dist_m=float(v['anchor_dist_m'])) for b, v in nm.items()}
dec_name = {b: v['identity'] for b, v in NAME['dec'].items() if v['in_vocab']}
apr_name = {b: v['identity'] for b, v in NAME['apr'].items() if v['in_vocab']}; apr_name_out = {b: v['identity'] for b, v in NAME['apr'].items() if not v['in_vocab']}
bag_of = {'dec': {a: b for b, a in dec_name.items()}, 'apr': {a: b for b, a in apr_name.items()}}
assert len(bag_of['dec']) == len(dec_name) and len(bag_of['apr']) == len(apr_name), 'an identity names two bags'
summary = dict(test='JS-1 twin ledger', design=dict(jirs_commit='37e263f', received_commit='4ef2ab0', addendum_commit='baf5123', addendum2_commit='7023dad'), constants=dict(OBS_MIN=OBS_MIN, POS_R=POS_R, TOP_R=TOP_R, TOP_MIN=TOP_MIN, TOP_K=TOP_K, PAIR_R=PAIR_R, THR=THR, T1_SHARE=T1_SHARE),
               lane_constant=LANE, cells={s: CELLS[s] for s in SEASONS}, left_out=P['apr']['left_out'],
               naming={s: dict(census_bags=int(P[s]['n_census']), anchors=int(P[s]['n_anchors']), vocabulary=len(P[s]['vocab']), named=len(NAME[s]), named_in_vocab=sum(v['in_vocab'] for v in NAME[s].values()),
                               named_outside_vocab=sum(not v['in_vocab'] for v in NAME[s].values()), vote_agrees=sum(v['vote_agrees'] is True and v['in_vocab'] for v in NAME[s].values()),
                               vote_disagrees=sum(v['vote_agrees'] is False and v['in_vocab'] for v in NAME[s].values()), no_vote=sum(v['vote_agrees'] is None and v['in_vocab'] for v in NAME[s].values())) for s in SEASONS})

# ------------------------------------------------------------------ T-id1 position and T-id3 height, per season and method
T1, T3 = {}, {}; dist = {s: {m: {} for m in ('b', 'a')} for s in SEASONS}
for s in SEASONS:
    T1[s], T3[s] = {}, {}; named = sorted(b for b, v in NAME[s].items() if v['in_vocab']); hl = HZ[s]['height_cam'].astype(float)
    for m in ('b', 'a'):
        for sub, bags in (('all', named), ('vote_agrees', [b for b in named if NAME[s][b]['vote_agrees'] is True])):
            d, unread = [], Counter()
            for b in bags:
                r = R[s][m][NAME[s][b]['identity']]
                if 'pos' not in r: unread[r['reason']] += 1; continue
                dd = float(np.hypot(r['pos'][0] - BXYZ[s][b, 0], r['pos'][2] - BXYZ[s][b, 2])); d.append(dd)
                if sub == 'all': dist[s][m][b] = dd
            within = int(np.sum(np.array(d) <= PAIR_R)) if d else 0
            T1[s][f'{m}_{sub}'] = dict(named=len(bags), read=len(d), unread=int(sum(unread.values())), unread_reasons=dict(unread), within_0_5=within, share_within_0_5=(within / len(bags) if bags else None), **{f'dist_{k}': v for k, v in stats_pos(d).items() if k != 'n'})
        for line in ('height', 'height_lidarlane', 'height_lanecut'):
            diff, excl = [], Counter()
            for b in named:
                r = R[s][m][NAME[s][b]['identity']]
                if line not in r: excl['no twin position' if 'pos' not in r else 'no twin height'] += 1; continue
                if not np.isfinite(hl[b]): excl['lidar height_cam not finite'] += 1; continue
                diff.append(r[line] - hl[b])
            T3[s][f'{m}_{line}'] = dict(named=len(named), excluded=dict(excl), **stats_abs(diff))
    log(f'T-id1 {s}: ' + '; '.join(f'{k}: {v["within_0_5"]}/{v["named"]} within 0.5 m, median {v.get("dist_median")}' for k, v in T1[s].items()))
    log(f'T-id3 {s}: ' + '; '.join(f'{k}: n {v["n"]} p95 {v.get("abs_p95")}' for k, v in T3[s].items()))

# ------------------------------------------------------------------ T-id2 found again (method b)
twinP = {}
for s in SEASONS:
    addr = sorted(a for a, r in R[s]['b'].items() if 'pos' in r); X3 = np.array([R[s]['b'][a]['pos'] for a in addr], float)
    if s == 'apr': X3 = X3 @ G[:3, :3].T + G[:3, 3]
    twinP[s] = (addr, X3)
ti, tj, tdist = mutual_nn(twinP['dec'][1][:, [0, 2]], twinP['apr'][1][:, [0, 2]])
TP = [(twinP['dec'][0][i], twinP['apr'][0][j], float(d)) for i, j, d in zip(ti, tj, tdist)]
tp_dec = {a: (b, d) for a, b, d in TP}; tp_apr = {b: (a, d) for a, b, d in TP}; pairs_out = []
cat = Counter(); both_cat = Counter(); recov = 0
for d_, a_, dd in zip(LD, LA, LDIST):
    d_, a_ = int(d_), int(a_); dn, an, ano = d_ in dec_name, a_ in apr_name, a_ in apr_name_out
    status = 'both named' if (dn and an) else ('one named' if (dn or an) else 'none named'); cat[status] += 1
    if ano: cat[f'{status}; April bag named outside the vocabulary'] += 1
    rec = dict(kind='lidar_pair', dec_bag=d_, apr_bag=a_, dist=float(dd), status=status, A_dec=dec_name.get(d_), A_apr=apr_name.get(a_, apr_name_out.get(a_)), apr_named_outside_vocab=bool(ano), L1=L1[d_], L2=L2.get(d_))
    if status == 'both named':
        Ad, Aa = dec_name[d_], apr_name[a_]; pd_, pa_ = tp_dec.get(Ad), tp_apr.get(Aa)
        if pd_ is not None and pd_[0] == Aa: c_ = 'same pair'; recov += 1
        elif pd_ is not None or pa_ is not None: c_ = 'paired elsewhere (both)' if (pd_ is not None and pa_ is not None) else 'paired elsewhere (one)'
        else:
            hp = int('pos' in R['dec']['b'][Ad]) + int('pos' in R['apr']['b'][Aa]); c_ = 'neither paired (' + {2: 'both read', 1: 'one read', 0: 'neither read'}[hp] + ')'
        both_cat[c_] += 1; rec['twin'] = c_
    pairs_out.append(rec)
twin_only = 0; twin_unnamed = 0; tp_cat = Counter()
for Ad, Aa, dd in TP:
    bd, ba = bag_of['dec'].get(Ad), bag_of['apr'].get(Aa)
    if bd is None or ba is None: c_ = 'address names no census bag'; twin_unnamed += 1
    elif (bd, ba) in lidar_pair_set: c_ = 'same pair'
    else: c_ = 'twin only'; twin_only += 1
    tp_cat[c_] += 1
    hdd, haa = R['dec']['b'][Ad].get('height'), R['apr']['b'][Aa].get('height'); chg = (haa - hdd) if (hdd is not None and haa is not None) else None
    pairs_out.append(dict(kind='twin_pair', A_dec=Ad, A_apr=Aa, dist=dd, dec_bag=bd, apr_bag=ba, category=c_, h_dec=hdd, h_apr=haa, change=chg, call=(call_of(chg) if chg is not None else None), L2=(L2.get(bd) if bd is not None else None), L1=(L1.get(bd) if bd is not None else None)))
nboth = cat['both named']
T2 = dict(twin_plants=dict(dec=len(twinP['dec'][0]), apr=len(twinP['apr'][0])), twin_pairs=len(TP), twin_pair_dist=stats_pos([t[2] for t in TP]), twin_pair_categories=dict(tp_cat),
          lidar_pairs=len(LD), lidar_pair_status=dict(cat), lidar_pairs_both_named=dict(n=nboth, **dict(both_cat)), same_pair=recov, twin_only=twin_only, twin_pairs_with_unnamed_address=twin_unnamed,
          recoverable_of_both_named=(recov / nboth if nboth else None), recoverable_of_all_lidar_pairs=recov / len(LD))
log(f'T-id2: twin pairs {len(TP)}; lidar pairs 680, both named {nboth}; same pair {recov}; twin only {twin_only}; categories {dict(both_cat)}')

# ------------------------------------------------------------------ T-id4 growth (method b)
S_all = [p for p in pairs_out if p['kind'] == 'twin_pair' and p['change'] is not None and p['dec_bag'] is not None and p['L2'] is not None]
S_same = [p for p in S_all if p['category'] == 'same pair']
T4 = {}
for name, S_ in (('all_twin_pairs_with_paper_call', S_all), ('same_pairs_of_T_id2', S_same)):
    l1 = [p for p in S_ if p['L1'] is not None and p['L1']['call'] is not None]
    blk = dict(n=len(S_), twin_vs_L2=confusion([p['call'] for p in S_], [p['L2'] for p in S_]), L1_available=len(l1), L1_vs_L2=confusion([p['L1']['call'] for p in l1], [p['L2'] for p in l1]),
               twin_vs_L2_on_L1_subset=confusion([p['call'] for p in l1], [p['L2'] for p in l1]), twin_calls=dict(Counter(p['call'] for p in S_)), L2_calls=dict(Counter(p['L2'] for p in S_)))
    if name == 'same_pairs_of_T_id2':
        blk['twin_change_minus_L1_change'] = stats_abs([p['change'] - p['L1']['change'] for p in l1])
        blk['twin_change'] = stats_abs([p['change'] for p in S_]); blk['L1_change_same_pairs'] = stats_abs([p['L1']['change'] for p in l1])
    T4[name] = blk
twin_pairs_with_change = sum(1 for p in pairs_out if p['kind'] == 'twin_pair' and p['change'] is not None)
T4['counts'] = dict(twin_pairs=len(TP), with_both_twin_heights=twin_pairs_with_change, dec_address_names_no_bag=sum(1 for p in pairs_out if p['kind'] == 'twin_pair' and p['change'] is not None and p['dec_bag'] is None),
                    dec_bag_without_paper_call=sum(1 for p in pairs_out if p['kind'] == 'twin_pair' and p['change'] is not None and p['dec_bag'] is not None and p['L2'] is None))
log(f'T-id4: S_all {len(S_all)} agreement {T4["all_twin_pairs_with_paper_call"]["twin_vs_L2"]["agreement"]}; S_same {len(S_same)} agreement {T4["same_pairs_of_T_id2"]["twin_vs_L2"]["agreement"]}; L1 vs L2 {T4["all_twin_pairs_with_paper_call"]["L1_vs_L2"]["agreement"]}')

# ------------------------------------------------------------------ repeats (method b): split half and cross cell
REP = {}; split_out, cross_out = [], []
for s in SEASONS:
    dp, dh_ = [], []
    for a, r in R[s]['b'].items():
        if 'split' not in r: continue
        h1, h2 = r['split']; e = dict(season=s, address=a, n_obs_kf=r['n_obs_kf'], half=[{k: h.get(k) for k in ('n_obs_kf', 'pos', 'height', 'reason', 'height_reason')} for h in (h1, h2)])
        if 'pos' in h1 and 'pos' in h2: e['dpos'] = float(np.hypot(h1['pos'][0] - h2['pos'][0], h1['pos'][2] - h2['pos'][2])); dp.append(e['dpos'])
        if 'height' in h1 and 'height' in h2: e['dh'] = float(h1['height'] - h2['height']); dh_.append(e['dh'])
        split_out.append(e)
    cp, ch = [], []
    for a, r in R[s]['b'].items():
        pc = {c: x for c, x in r.get('per_cell', {}).items() if 'pos' in x}
        for i, c1 in enumerate(sorted(pc)):
            for c2 in sorted(pc)[i + 1:]:
                x1, x2 = pc[c1], pc[c2]; e = dict(season=s, address=a, cells=[c1, c2], n_obs_kf=[x1['n_obs_kf'], x2['n_obs_kf']], dpos=float(np.hypot(x1['pos'][0] - x2['pos'][0], x1['pos'][2] - x2['pos'][2])))
                cp.append(e['dpos'])
                if 'height' in x1 and 'height' in x2: e['dh'] = float(x1['height'] - x2['height']); ch.append(e['dh'])
                cross_out.append(e)
    REP[s] = dict(split_half=dict(position=stats_pos(dp), height=stats_abs(dh_)), cross_cell=dict(position=stats_pos(cp), height=stats_abs(ch)))
    log(f'repeats {s}: split half pos n {len(dp)} median {REP[s]["split_half"]["position"].get("median")}, height p95 {REP[s]["split_half"]["height"].get("abs_p95")}; cross cell pos n {len(cp)}, height n {len(ch)}')
LM = json.load(open(L / 'ledger_matrix.json'))['routes']['global']['pairs']; HR = json.load(open(L / 'heights_repeatability.json'))['pairs']; WED = ['apr1054', 'apr1130', 'apr1400', 'apr1517']
paper_same_day = {k: dict(position=dict(matched=v['matched'], median_m=v['median_m'], p90_m=v['p90_m']), height_cam=HR[k]['height_cam']) for k, v in LM.items() if all(x in WED for x in k.split('|'))}

# ------------------------------------------------------------------ verdict by the design's rule (method b, twin lane constant)
t1p = {s: T1[s]['b_all']['share_within_0_5'] >= T1_SHARE for s in SEASONS}; t3p = {s: (T3[s]['b_height']['n'] > 0 and T3[s]['b_height']['abs_p95'] <= THR) for s in SEASONS}
if all(t1p.values()) and all(t3p.values()): reading = 'The ledger can be read through the identity field (T-id1 and T-id3 pass in both seasons).'
elif all(t1p.values()): reading = 'Identity and position can be read, heights cannot at the paper\'s threshold (T-id1 passes in both seasons; T-id3 fails in ' + ' and '.join(s for s in SEASONS if not t3p[s]) + ').'
else: reading = 'The ledger cannot be read through the identity field as built in ' + ' and '.join({'dec': 'December', 'apr': 'April'}[s] for s in SEASONS if not t1p[s]) + ' (T-id1 fails there).'
VERDICT = dict(T_id1_pass=t1p, T_id3_pass=t3p, reading=reading)
log('VERDICT', json.dumps(VERDICT))

# ------------------------------------------------------------------ panels (ADDENDUM 12), chosen by rank before any is looked at
PANELS = []
for s in SEASONS:
    hl, top_l = HZ[s]['height_cam'].astype(float), HZ[s]['top'].astype(float); rows = []
    for b, v in NAME[s].items():
        if not v['in_vocab']: continue
        r = R[s]['b'][v['identity']]
        if 'height' in r and np.isfinite(hl[b]): rows.append((abs(r['height'] - hl[b]), b))
    rows.sort(); n = len(rows); A, K, C = RAW[s]
    for qq in (0.05, 0.25, 0.50, 0.75, 0.95, 1.0):
        i = int(np.floor(qq * (n - 1) + 0.5)); ad, b = rows[i]; v = NAME[s][b]; r = R[s]['b'][v['identity']]; kk = r['top_kf'][0]
        cells_here = sorted(set(int(c) for c in C[(A == v['identity']) & (K == kk)])); cell = CELLS[s][cells_here[0]]
        PANELS.append(dict(season=s, rank_q=qq, index=i, n=n, bag=int(b), address=int(v['identity']), abs_diff=float(ad), twin_height=r['height'], lidar_height=float(hl[b]), twin_top=r['top'], lidar_top=float(top_l[b]),
                           twin_pos=r['pos'], census_xyz=BXYZ[s][b].tolist(), keyframe=int(kk), cell=cell, cells_with_points=[CELLS[s][c] for c in cells_here]))
json.dump(PANELS, open(REC / 'panels_selected.json', 'w'), indent=1)

# ------------------------------------------------------------------ records
for s in SEASONS:
    with open(REC / f'plants_{s}.jsonl', 'w') as f:
        hl, top_l = HZ[s]['height_cam'].astype(float), HZ[s]['top'].astype(float)
        for a in [int(v) for v in P[s]['vocab']]:
            b = bag_of[s].get(a); e = dict(season=s, address=a, in_vocab=True, named_bag=b)
            if b is not None:
                e.update(census_xyz=BXYZ[s][b].tolist(), lidar_height_cam=(float(hl[b]) if np.isfinite(hl[b]) else None), lidar_top=(float(top_l[b]) if np.isfinite(top_l[b]) else None),
                         vote_agrees=NAME[s][b]['vote_agrees'], anchor_dist_m=NAME[s][b]['anchor_dist_m'], dist_b=dist[s]['b'].get(b), dist_a=dist[s]['a'].get(b))
            for m in ('b', 'a'):
                r = dict(R[s][m][a]); r.pop('split', None); pc = r.pop('per_cell', None); e[m] = r
                if pc is not None: e['b_per_cell'] = {c: {k: x.get(k) for k in ('n_obs_kf', 'pos', 'height', 'reason', 'height_reason')} for c, x in pc.items()}
            f.write(json.dumps(e) + '\n')
        for b, v in sorted(NAME[s].items()):
            if not v['in_vocab']: f.write(json.dumps(dict(season=s, address=v['identity'], in_vocab=False, named_bag=b, census_xyz=BXYZ[s][b].tolist(), lidar_height_cam=(float(hl[b]) if np.isfinite(hl[b]) else None), reason='identity outside the vocabulary of the used cells')) + '\n')
with open(REC / 'pairs.jsonl', 'w') as f:
    for p in pairs_out: f.write(json.dumps(p) + '\n')
with open(REC / 'split_half.jsonl', 'w') as f:
    for e in split_out: f.write(json.dumps(e) + '\n')
with open(REC / 'cross_cell.jsonl', 'w') as f:
    for e in cross_out: f.write(json.dumps(e) + '\n')
summary.update(T_id1=T1, T_id2=T2, T_id3=T3, T_id4=T4, repeats=REP, paper_same_day_lidar=paper_same_day, paper_dec_apr1054_lidar=dict(pairs=LM['dec|apr1054'], height_cam=HR['dec|apr1054']['height_cam'], beyond=HR['dec|apr1054'].get('beyond_threshold')),
               L2_paper_call=dict(n=len(L2), **dict(L2c)), verdict=VERDICT, panels=PANELS, written=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
json.dump(summary, open(RES / 'summary.json', 'w'), indent=1)
log('wrote', REC, RES / 'summary.json')
