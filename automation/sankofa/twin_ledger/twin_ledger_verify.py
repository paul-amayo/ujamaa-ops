#!/usr/bin/env python3
"""twin_ledger_verify.py (11 Nov 2026, A100 session) - JIRS test JS-1, step 4 of 5: an independent recount of every table of
summary.json. Written separately from twin_ledger_score.py and sharing no code with it: pairs by brute-force distance matrices
(not KD-trees); the paper's majority call (L2) re-implemented from make_figures.py's description; lidar positions and heights
read from the npz files, naming from prep.json; only the per-address twin records (position, height) are taken from the
jsonl records. Then a sample of those records (30 random addresses per season and every panel address, method b and a) is
re-derived from the raw point files with its own grouping code. Prints every comparison; exits 1 on any mismatch.
  python3 twin_ledger_verify.py <prep.json> <points dir> <records dir> <results dir> <lane dir>"""
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

PREP, PTS, REC, RES, LANED = (Path(x) for x in sys.argv[1:6])
P = json.load(open(PREP)); L = Path(P['census_dir']); SUM = json.load(open(RES / 'summary.json'))
TH, R_ASSOC = 0.082, 0.5; BAD = []


def chk(name, mine, theirs, tol=1e-9):
    if isinstance(mine, (int, float, np.floating, np.integer)) and isinstance(theirs, (int, float)) and theirs is not None:
        ok = abs(float(mine) - float(theirs)) <= tol
    else:
        ok = mine == theirs
    print(f"{'OK ' if ok else 'BAD'} {name}: recount {mine} | summary {theirs}")
    if not ok: BAD.append(name)


def mnn(A, B, r=R_ASSOC):
    """mutual nearest neighbours within r by a full distance matrix; non-finite rows skipped; original indices"""
    fa = [i for i in range(len(A)) if np.all(np.isfinite(A[i]))]; fb = [j for j in range(len(B)) if np.all(np.isfinite(B[j]))]
    D = np.sqrt(((A[fa][:, None, :] - B[fb][None, :, :]) ** 2).sum(-1)); ja = D.argmin(1); ib = D.argmin(0)
    return [(fa[i], fb[ja[i]], float(D[i, ja[i]])) for i in range(len(fa)) if D[i, ja[i]] <= r and ib[ja[i]] == i]


def lab(c): return 'grown' if c > TH else ('shrunk' if c < -TH else 'stable')


def pct(x, p): x = np.sort(np.asarray(x, float)); return float(np.quantile(x, p / 100.0))


# ---------------- inputs read independently
cen = {'dec': np.load(L / 'census_dec.npz'), 'apr': np.load(L / 'census_apr1054.npz')}
hz = {'dec': np.load(L / 'heights_dec.npz'), 'apr': np.load(L / 'heights_apr1054.npz')}
pl54 = np.load(L / 'placed_apr1054.npz'); G = pl54['G_global']
plants = {s: [json.loads(l) for l in open(REC / f'plants_{s}.jsonl')] for s in ('dec', 'apr')}
pairs = [json.loads(l) for l in open(REC / 'pairs.jsonl')]
rec = {s: {e['address']: e for e in plants[s] if e['in_vocab']} for s in plants}
naming = {s: {int(b): v for b, v in P[s]['naming'].items()} for s in ('dec', 'apr')}

# ---------------- lane constants from the raw point files
for s in ('dec', 'apr'):
    lh = np.concatenate([np.load(LANED / f'{s}_{c}.npz')['h_nocut'] for c in sorted(P[s]['cells'])]).astype(float)
    chk(f'{s} lane constant (design rule, no opacity cut)', float(np.median(lh)), SUM['lane_constant'][s]['twin'], 1e-9)
    lc = np.concatenate([np.load(PTS / f'{s}_{c}.npz')['lane_h'] for c in sorted(P[s]['cells'])]).astype(float)
    chk(f'{s} lane constant (render, opacity cut)', float(np.median(lc)), SUM['lane_constant'][s]['opacity_cut'], 1e-9)

# ---------------- T-id1 and T-id3 from the records + census/heights files
for s in ('dec', 'apr'):
    named = sorted(b for b, v in naming[s].items() if v['in_vocab']); B = cen[s]['bags_xyz'].astype(float); H = hz[s]['height_cam'].astype(float)
    chk(f'{s} named in vocab', len(named), SUM['naming'][s]['named_in_vocab'])
    for m in ('b', 'a'):
        for sub in ('all', 'vote_agrees'):
            bags = named if sub == 'all' else [b for b in named if naming[s][b]['vote_agrees'] is True]
            d = []; un = 0
            for b in bags:
                r = rec[s][naming[s][b]['identity']][m]
                if r.get('pos') is None: un += 1; continue
                d.append(float(np.sqrt((r['pos'][0] - B[b, 0]) ** 2 + (r['pos'][2] - B[b, 2]) ** 2)))
            T = SUM['T_id1'][s][f'{m}_{sub}']; w = sum(x <= R_ASSOC for x in d)
            chk(f'T1 {s} {m} {sub} named', len(bags), T['named']); chk(f'T1 {s} {m} {sub} unread', un, T['unread']); chk(f'T1 {s} {m} {sub} within 0.5', w, T['within_0_5'])
            if d: chk(f'T1 {s} {m} {sub} median', float(np.median(d)), T['dist_median'], 1e-9); chk(f'T1 {s} {m} {sub} p95', pct(d, 95), T['dist_p95'], 1e-9)
            chk(f'T1 {s} {m} {sub} share', w / len(bags), T['share_within_0_5'], 1e-12)
        for line in ('height', 'height_lidarlane', 'height_lanecut'):
            dd = []
            for b in named:
                r = rec[s][naming[s][b]['identity']][m]
                if r.get(line) is None or not np.isfinite(H[b]): continue
                dd.append(r[line] - H[b])
            T = SUM['T_id3'][s][f'{m}_{line}']; chk(f'T3 {s} {m} {line} n', len(dd), T['n'])
            if dd:
                chk(f'T3 {s} {m} {line} bias', float(np.median(dd)), T['bias_median'], 1e-9); chk(f'T3 {s} {m} {line} |d| median', float(np.median(np.abs(dd))), T['abs_median'], 1e-9)
                chk(f'T3 {s} {m} {line} |d| p95', pct(np.abs(dd), 95), T['abs_p95'], 1e-9)

# ---------------- T-id2 by brute force
addr = {s: sorted(a for a, e in rec[s].items() if e['b'].get('pos') is not None) for s in rec}
Xd = np.array([rec['dec'][a]['b']['pos'] for a in addr['dec']]); Xa = np.array([rec['apr'][a]['b']['pos'] for a in addr['apr']])
Xa = (G[:3, :3] @ Xa.T).T + G[:3, 3]
tw = [(addr['dec'][i], addr['apr'][j], d) for i, j, d in mnn(Xd[:, [0, 2]], Xa[:, [0, 2]])]
chk('T2 twin pairs', len(tw), SUM['T_id2']['twin_pairs']); chk('T2 twin plants dec', len(addr['dec']), SUM['T_id2']['twin_plants']['dec']); chk('T2 twin plants apr', len(addr['apr']), SUM['T_id2']['twin_plants']['apr'])
mine_tw = sorted((a, b) for a, b, _ in tw); theirs_tw = sorted((p['A_dec'], p['A_apr']) for p in pairs if p['kind'] == 'twin_pair'); chk('T2 twin pair list identical', mine_tw == theirs_tw, True)
lp = mnn(cen['dec']['bags_xyz'].astype(float)[:, [0, 2]], pl54['global'].astype(float)[:, [0, 2]]); chk('T2 lidar pairs', len(lp), 680)
dn = {b: v['identity'] for b, v in naming['dec'].items() if v['in_vocab']}; an = {b: v['identity'] for b, v in naming['apr'].items() if v['in_vocab']}
tpd = {a: b for a, b, _ in tw}; tpa = {b: a for a, b, _ in tw}; both = same = 0; cats = Counter(); status = Counter()
for d, a, _ in lp:
    st = 'both named' if (d in dn and a in an) else ('one named' if (d in dn or a in an) else 'none named'); status[st] += 1
    if st != 'both named': continue
    both += 1; Ad, Aa = dn[d], an[a]
    if tpd.get(Ad) == Aa: same += 1; cats['same pair'] += 1
    elif Ad in tpd or Aa in tpa: cats['paired elsewhere'] += 1
    else: cats['neither paired'] += 1
T2 = SUM['T_id2']; chk('T2 both named', both, T2['lidar_pairs_both_named']['n']); chk('T2 same pair', same, T2['same_pair'])
for k_ in ('one named', 'none named'): chk(f'T2 lidar pairs {k_}', status[k_], T2['lidar_pair_status'].get(k_, 0))
chk('T2 paired elsewhere', cats['paired elsewhere'], sum(v for k, v in T2['lidar_pairs_both_named'].items() if k.startswith('paired elsewhere')))
chk('T2 neither paired', cats['neither paired'], sum(v for k, v in T2['lidar_pairs_both_named'].items() if k.startswith('neither paired')))
lps = {(d, a) for d, a, _ in lp}; bd_of = {v: k for k, v in dn.items()}; ba_of = {v: k for k, v in an.items()}
to = sum(1 for Ad, Aa, _ in tw if Ad in bd_of and Aa in ba_of and (bd_of[Ad], ba_of[Aa]) not in lps); chk('T2 twin only', to, T2['twin_only'])
chk('T2 twin pairs with an unnamed address', sum(1 for Ad, Aa, _ in tw if Ad not in bd_of or Aa not in ba_of), T2['twin_pairs_with_unnamed_address'])

# ---------------- T-id4: L2 re-implemented, L1, twin calls
DXZ = cen['dec']['bags_xz'].astype(float); hdec = hz['dec']['height_cam'].astype(float); vt = defaultdict(lambda: [0, 0, 0])
for p in ('apr1130', 'apr1400', 'apr16_1230'):
    pp = np.load(L / f'placed_{p}.npz')['global'].astype(float)[:, [0, 2]]; hp = np.load(L / f'heights_{p}.npz')['height_cam'].astype(float)
    for d, a, _ in mnn(DXZ, pp):
        c = hp[a] - hdec[d]
        if np.isfinite(c): vt[d][0 if c > TH else (2 if c < -TH else 1)] += 1
L2 = {d: ['grown', 'stable', 'shrunk'][int(np.argmax(v))] for d, v in vt.items()}
chk('L2 bags called', len(L2), 728); chk('L2 grown/stable/shrunk', [sum(1 for v in L2.values() if v == c) for c in ('grown', 'stable', 'shrunk')], [227, 366, 135])
hapr = hz['apr']['height_cam'].astype(float); L1 = {d: hapr[a] - hdec[d] for d, a, _ in lp}
Sall, Ssame = [], []
for Ad, Aa, _ in tw:
    hd_, ha_ = rec['dec'][Ad]['b'].get('height'), rec['apr'][Aa]['b'].get('height')
    if hd_ is None or ha_ is None or Ad not in bd_of or bd_of[Ad] not in L2: continue
    e = (Ad, Aa, ha_ - hd_, bd_of[Ad]); Sall.append(e)
    if Aa in ba_of and (bd_of[Ad], ba_of[Aa]) in lps: Ssame.append(e)
for nm, S_ in (('all_twin_pairs_with_paper_call', Sall), ('same_pairs_of_T_id2', Ssame)):
    T = SUM['T_id4'][nm]; M = np.zeros((3, 3), int); idx = {'grown': 0, 'stable': 1, 'shrunk': 2}
    for _, _, c, d in S_: M[idx[lab(c)], idx[L2[d]]] += 1
    chk(f'T4 {nm} n', len(S_), T['n']); chk(f'T4 {nm} twin vs L2 matrix', M.tolist(), T['twin_vs_L2']['matrix'])
    l1 = [(c, d) for _, _, c, d in S_ if d in L1 and np.isfinite(L1[d])]; M1 = np.zeros((3, 3), int)
    for _, d in l1: M1[idx[lab(L1[d])], idx[L2[d]]] += 1
    chk(f'T4 {nm} L1 available', len(l1), T['L1_available']); chk(f'T4 {nm} L1 vs L2 matrix', M1.tolist(), T['L1_vs_L2']['matrix'])
    if nm == 'same_pairs_of_T_id2':
        dd = [c - L1[d] for c, d in l1]; X = T['twin_change_minus_L1_change']
        chk('T4 same twin-L1 n', len(dd), X['n']); chk('T4 same twin-L1 bias', float(np.median(dd)), X['bias_median'], 1e-9); chk('T4 same twin-L1 |d| p95', pct(np.abs(dd), 95), X['abs_p95'], 1e-9)

# ---------------- repeats from their records
for s in ('dec', 'apr'):
    sp = [json.loads(l) for l in open(REC / 'split_half.jsonl')]; sp = [e for e in sp if e['season'] == s]
    dp = [e['dpos'] for e in sp if 'dpos' in e]; dh = [e['dh'] for e in sp if 'dh' in e]; T = SUM['repeats'][s]['split_half']
    chk(f'split {s} pos n', len(dp), T['position']['n']); chk(f'split {s} pos median', float(np.median(dp)), T['position']['median'], 1e-9)
    chk(f'split {s} h n', len(dh), T['height']['n']); chk(f'split {s} h p95', pct(np.abs(dh), 95), T['height']['abs_p95'], 1e-9)
    cc = [json.loads(l) for l in open(REC / 'cross_cell.jsonl')]; cc = [e for e in cc if e['season'] == s]; T = SUM['repeats'][s]['cross_cell']
    cp = [e['dpos'] for e in cc]; ch = [e['dh'] for e in cc if 'dh' in e]
    chk(f'cross {s} pos n', len(cp), T['position']['n'])
    if cp: chk(f'cross {s} pos median', float(np.median(cp)), T['position']['median'], 1e-9)
    chk(f'cross {s} h n', len(ch), T['height']['n'])
    if ch: chk(f'cross {s} h p95', pct(np.abs(ch), 95), T['height']['abs_p95'], 1e-9)

# ---------------- verdict by the rule
t1 = {s: SUM['T_id1'][s]['b_all']['within_0_5'] / SUM['T_id1'][s]['b_all']['named'] >= 0.95 for s in ('dec', 'apr')}
t3 = {s: SUM['T_id3'][s]['b_height']['abs_p95'] <= TH for s in ('dec', 'apr')}
chk('verdict T-id1', t1, SUM['verdict']['T_id1_pass']); chk('verdict T-id3', t3, SUM['verdict']['T_id3_pass'])

# ---------------- re-derive a sample of records from the raw points (own grouping code)
rng = np.random.default_rng(20261111)
for s in ('dec', 'apr'):
    up = np.array(P[s]['up'], float); camh = {int(k): float(np.dot(v, up)) for k, v in P[s]['cam_centres_graph'].items()}; lane = SUM['lane_constant'][s]['twin']
    vocab = sorted(rec[s]); pick = sorted(set(rng.choice(vocab, 30, replace=False).tolist()) | {p['address'] for p in SUM['panels'] if p['season'] == s})
    for m in ('b', 'a'):
        byk = {a: defaultdict(list) for a in pick}
        for c in sorted(P[s]['cells']):
            z = np.load(PTS / f'{s}_{c}.npz'); ad = z[f'{m}_addr']; sel = np.isin(ad, pick)
            for x, a, k in zip(z[f'{m}_xyz'][sel].astype(np.float64), ad[sel], z[f'{m}_k'][sel]): byk[int(a)][int(k)].append(x)
        nbad = 0
        for a in pick:
            obs = {k: np.array(v) for k, v in byk[a].items() if len(v) >= 10}; r = rec[s][a][m]
            if not obs:
                ok = r.get('pos') is None
            else:
                Q = np.vstack([obs[k] for k in sorted(obs)]); cx, cz = np.median(Q[:, 0]), np.median(Q[:, 2])
                S_ = Q[np.sqrt((Q[:, 0] - cx) ** 2 + (Q[:, 2] - cz) ** 2) <= 0.35]
                pos = [float(np.median(S_[:, i])) for i in range(3)] if len(S_) else None
                tops, chs = [], []
                if pos is not None:
                    for k in sorted(obs, key=lambda kk: (len(obs[kk]), kk), reverse=True)[:12]:
                        V = obs[k]; col = V[np.sqrt((V[:, 0] - pos[0]) ** 2 + (V[:, 2] - pos[2]) ** 2) <= 0.30]
                        if len(col) >= 20: tops.append(float(np.quantile(col @ up, 0.95))); chs.append(camh[k])
                h = (float(np.median(tops)) - (float(np.median(chs)) - lane)) if tops else None
                ok = (pos is None and r.get('pos') is None) or (pos is not None and r.get('pos') is not None and np.allclose(pos, r['pos'], atol=1e-7))
                ok &= (h is None and r.get('height') is None) or (h is not None and r.get('height') is not None and abs(h - r['height']) < 1e-7)
            nbad += (not ok)
            if not ok: print(f'BAD record {s} {m} address {a}: re-derived differs')
        chk(f're-derived records {s} {m} ({len(pick)} addresses), mismatches', nbad, 0)
print(f'\n[verify] {len(BAD)} mismatches' + (f': {BAD}' if BAD else ''))
sys.exit(1 if BAD else 0)
