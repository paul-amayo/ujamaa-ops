#!/usr/bin/env python3
"""azalai_online_verify.py (11 Nov 2026) — RSS Azalai pilot AZ-2: independent verification of azalai_online_score.py for one
(registry, revisit) pair. Shares no code with the scorer.
  1. RECOUNT: every count, rate, interval and sign test of online_summary_<tag>.json recomputed from online_targets_<tag>.jsonl.
  2. RE-DECISION of the arrivals without the lane frames: at each recorded stop (stop_k, stop_frac) the robot's graph position is
     interpolated, the local travel direction is taken from the graph track (7 keyframes about the stop), the aim point is d ahead,
     and the nearest revisit census bag on the partner's side of the track (within 1.5 m of it) along the travel direction is the
     arrival. Compared with the recorded arrival for every variant; disagreements are listed.
  3. COUNTING re-derived without the lane frames: bags seen from the lane pass within 1.5 m of the line through its camera
     centres, ordered along that line (not the lane frame's axis), per side; the registry rank n and the revisit's n-th bag.
  4. CAUSAL BAGS against the whole-pass census (graph poses, last keyframe of each lane): how many inner-row census bags have a
     causal bag within 0.2 m and the reverse; and at the row end, whether the first causal inner-row bag is the census's first.
  5. TIME ALIGNMENT of the laser odometry: the lag (ms) that best correlates its speed profile with the graph's, per pass.
nerf_new python3.10:  azalai_online_verify.py --reg dec --rev apr1130 --records <dir> --causal <dir> [--direct]"""
import argparse, json, math, numpy as np
from pathlib import Path
from scipy.stats import beta, binom
ap = argparse.ArgumentParser(); ap.add_argument('--reg', required=True); ap.add_argument('--rev', required=True); ap.add_argument('--records', required=True); ap.add_argument('--causal', required=True); ap.add_argument('--direct', action='store_true'); a = ap.parse_args()
TAG = f'{a.reg}_{a.rev}' + ('_direct' if a.direct else ''); RD = Path(a.records); S = json.load(open(RD / f'online_summary_{TAG}.json')); R = [json.loads(l) for l in open(RD / f'online_targets_{TAG}.jsonl')]
X = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental'); SK = X / 'sankofa/jirs_20261023/lidar_fixedreader'; EXP = json.load(open(X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'))['experiences']
bad = []
def chk(name, got, want):
    ok = (got == want) if not isinstance(got, float) else (want is not None and abs(got - want) < 1e-9)
    if not ok: bad.append((name, got, want))
def ci(k, n): return (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
def sign_p(x, y):
    n = x + y
    if n == 0: return 1.0
    pm = binom.pmf(np.arange(n + 1), n, 0.5); return float(min(1.0, pm[pm <= pm[x] * (1 + 1e-7)].sum()))
# ---------------------------------------------------------------- 1. recount
T = [r for r in R if 'skip' not in r]; SE = [r for r in T if r['stop_evaluable']]
chk('n_pairs', len(R), S['n_pairs']); chk('targets', len(T), S['targets']); chk('stop_evaluable', len(SE), S['stop_evaluable'])
nt = {}
for r in R:
    if 'skip' in r: nt[r['skip']] = nt.get(r['skip'], 0) + 1
chk('not_target', nt, S['not_target'])
def tab(rows, f):
    o = [f(r) for r in rows]; return dict(n=len(o), right=o.count('right'), neighbour=o.count('neighbour'), other=o.count('other'))
def cmp(name, rows, f, want):
    t = tab(rows, f)
    for k in ('n', 'right', 'neighbour', 'other'): chk(f'{name}.{k}', t[k], want[k])
    if t['n']:
        chk(f'{name}.rate', t['right'] / t['n'], want['rate']); lo, hi = ci(t['right'], t['n']); chk(f'{name}.ci_lo', float(lo), want['ci95'][0]); chk(f'{name}.ci_hi', float(hi), want['ci95'][1])
    return t
fc = lambda r: r['counting']['outcome']
cmp('counting.all', T, fc, S['counting']['all_targets']); cmp('counting.stop_evaluable', SE, fc, S['counting']['stop_evaluable'])
for band, want in S['counting']['by_n'].items():
    lo, hi = band.split('-'); lo = int(lo); hi = 10 ** 9 if hi == 'up' else int(hi); rows = [r for r in T if lo <= r['n'] <= hi]; cmp(f'counting.by_n.{band}', rows, fc, want)
    if rows: chk(f'counting.by_n.{band}.abs_offset_med', float(np.median([abs(r['counting']['count_offset']) for r in rows])), want['abs_offset_med'])
for ln, want in S['counting']['by_lane'].items(): cmp(f'counting.lane{ln}', [r for r in T if r['lane'] == int(ln)], fc, want)
print(f'[verify] {TAG}: {len(R)} pairs, {len(T)} targets, {len(SE)} stop-evaluable')
print('  counting (all targets):', tab(T, fc), '| by n:', {b: (tab([r for r in T if int(b.split("-")[0]) <= r['n'] <= (10 ** 9 if b.endswith('up') else int(b.split('-')[1]))], fc)) for b in S['counting']['by_n']})
for v, want in S['variants'].items():
    rows = [r for r in SE if 'outcome' in r['var'][v]]; f = lambda r, v=v: r['var'][v]['outcome']; t = cmp(f'var.{v}', rows, f, want); chk(f'var.{v}.no_stop', sum(r['var'][v]['no_stop'] for r in rows), want['no_stop'])
    for ln, w2 in want['by_lane'].items(): cmp(f'var.{v}.lane{ln}', [r for r in rows if r['lane'] == int(ln)], f, w2)
    for r in rows:   # right means the arrival is the partner, and only then
        if not r['var'][v]['no_stop']: chk(f'var.{v}.right_is_partner.{r["registry_bag"]}', (r['var'][v]['arrival'] == r['partner']), (r['var'][v]['outcome'] == 'right'))
    if t['n']: lo, hi = ci(t['right'], t['n']); print(f"  {v:4s} right {t['right']}/{t['n']} = {t['right'] / t['n']:.3f} [{lo:.3f}, {hi:.3f}]  neighbour {t['neighbour']}  other {t['other']}")
for key, want in S['paired'].items():
    x, y = key.split('_vs_'); g = lambda r, n: r['counting']['outcome'] if n == 'counting' else r['var'][n]['outcome']; rows = [r for r in SE if all(n == 'counting' or 'outcome' in r['var'][n] for n in (x, y))]
    fo = sum(g(r, x) == 'right' and g(r, y) != 'right' for r in rows); so = sum(g(r, x) != 'right' and g(r, y) == 'right' for r in rows)
    chk(f'paired.{key}.n', len(rows), want['n']); chk(f'paired.{key}.first_only', fo, want['first_only']); chk(f'paired.{key}.second_only', so, want['second_only']); p = sign_p(fo, so)
    if abs(p - want['p']) > 1e-9 * max(1.0, want['p']) and abs(math.log10(max(p, 1e-300)) - math.log10(max(want['p'], 1e-300))) > 1e-6: bad.append((f'paired.{key}.p', p, want['p']))
    print(f'  paired {key}: n {len(rows)}, first only {fo}, second only {so}, p = {p:.3g}')
print(f'[verify] recount: {"ALL AGREE with the summary" if not bad else f"{len(bad)} DISAGREEMENTS"}');
for b in bad[:40]: print('   ', b)
# ---------------------------------------------------------------- 2-3. arrivals and counting without the lane frames
def load(name):
    C = np.array(EXP[name]['c2w'], float); z = np.load(SK / f'census_{name}.npz'); return C, z['bags_xyz'].astype(float), z['obs_bag'], z['obs_k']
Cr, Br, ob_r, ok_r = load(a.rev); Ct, Bt, ob_t, ok_t = load(a.reg); D = S['standoff_d_m']; lanes = {l['lane']: l for l in S['lanes']}
def lane_rows(C, B, ob, okf, k0, k1, sense):
    """bags seen from keyframes k0..k1 within 1.5 m of the line through the camera centres -> {bag: (along the line in the driving sense, signed offset)}"""
    P = C[k0:k1 + 1][:, [0, 2], 3]; m = P.mean(0); u, s_, vt = np.linalg.svd(P - m); d = vt[0] * np.sign(vt[0] @ (P[-1] - P[0])); nrm = np.array([d[1], -d[0]]) * sense   # nrm: +x side for either sense
    out = {}
    for b in np.unique(ob[(okf >= k0) & (okf <= k1)]):
        if not np.isfinite(B[b]).all(): continue
        q = B[b][[0, 2]] - m; off = float(q @ nrm)
        if abs(off) <= 1.5: out[int(b)] = (float(q @ d), off)
    return out
# the registry's lane passes: found here from the summary's revisit lanes is not possible, so they are re-derived by the heading rule
def passes(C):
    z = C[:, 2, 2]; x = C[:, 0, 3]; fin = np.isfinite(C).all((1, 2)); out = []; st = None
    for k in range(len(C) + 1):
        s = 0 if (k == len(C) or not fin[k] or abs(z[k]) <= 0.8) else (1 if z[k] > 0 else -1)
        if st is not None and (s != st[1] or abs(x[k] - x[k - 1]) > 0.6):
            if k - st[0] >= 30: out.append((st[0], k - 1, st[1]))
            st = None
        if st is None and s != 0: st = (k, s)
    return out
Pt = passes(Ct); Pr = passes(Cr); assert [(l['k0'], l['k1'], l['sense']) for l in S['lanes']] == [tuple(p) for p in Pr], 'lane passes differ from the scorer'
rows_t = [lane_rows(Ct, Bt, ob_t, ok_t, *p) for p in Pt]; rows_r = [lane_rows(Cr, Br, ob_r, ok_r, *p) for p in Pr]
dis = {v: [] for v in S['variants']}; ncmp = {v: 0 for v in S['variants']}; cdis = []; ncnt = 0
for r in T:
    i = r['lane'] - 1; k0, k1, sense = Pr[i]; rr = rows_r[i]; rt = rows_t[i]; p = r['partner']; t = r['registry_bag']
    if p not in rr or t not in rt: cdis.append((t, 'not in the re-derived inner rows')); continue
    side = np.sign(rr[p][1]); same = sorted((al, b) for b, (al, off) in rr.items() if np.sign(off) == side)
    # counting
    n = 1 + sum(1 for b, (al, off) in rt.items() if np.sign(off) == np.sign(rt[t][1]) and al < rt[t][0]); nth = same[n - 1][1] if n <= len(same) else None; ncnt += 1
    if n != r['n'] or nth != r['counting']['arrival']: cdis.append((t, n, r['n'], nth, r['counting']['arrival']))
    if not r['stop_evaluable']: continue
    for v in S['variants']:
        o = r['var'][v]
        if 'outcome' not in o or o['no_stop']: continue
        j = o['stop_k']; f = o['stop_frac']; pos = Cr[j, :3, 3] if f >= 1.0 or j == k0 else (1 - f) * Cr[j - 1, :3, 3] + f * Cr[j, :3, 3]
        ja, jb = max(k0, j - 3), min(k1, j + 3); dirn = Cr[jb, [0, 2], 3] - Cr[ja, [0, 2], 3]; dirn /= np.linalg.norm(dirn); aim = pos[[0, 2]] + D * dirn
        P = Cr[k0:k1 + 1][:, [0, 2], 3]; m = P.mean(0); d = np.linalg.svd(P - m)[2][0]; d = d * np.sign(d @ (P[-1] - P[0])); al_aim = float((aim - m) @ d)
        arr = min(same, key=lambda x: abs(x[0] - al_aim))[1]; ncmp[v] += 1
        if arr != o['arrival']: dis[v].append((t, arr, o['arrival'], round(o['miss_m'], 2)))
print(f'[verify] counting re-derived without the lane frames: {ncnt} targets, {len(cdis)} differ', cdis[:10])
for v in S['variants']: print(f'[verify] arrivals re-decided without the lane frames, {v}: {ncmp[v]} stops, {len(dis[v])} differ', dis[v][:6])
# ---------------------------------------------------------------- 4. causal bags against the census (graph poses, last keyframe of each lane)
cz = np.load(Path(a.causal) / f'causal_{a.rev}_graph.npz', allow_pickle=True); tot = [0, 0, 0, 0]; first = []
for i, (k0, k1, sense) in enumerate(Pr):
    st = np.flatnonzero(cz['step_lane'] == i)[-1]; Xb = cz['bags_xyz'][cz['step_start'][st]:cz['step_start'][st] + cz['step_n'][st]]; P = Cr[k0:k1 + 1][:, [0, 2], 3]; m = P.mean(0); d = np.linalg.svd(P - m)[2][0]; d = d * np.sign(d @ (P[-1] - P[0])); nrm = np.array([d[1], -d[0]]) * sense
    off = (Xb[:, [0, 2]] - m) @ nrm; al = (Xb[:, [0, 2]] - m) @ d; inn = np.abs(off) <= 1.5; cb = np.array(sorted(rows_r[i])); cal = np.array([rows_r[i][b][0] for b in cb]); coff = np.array([rows_r[i][b][1] for b in cb])
    d1 = np.array([np.hypot(al[inn] - x, off[inn] - y).min() for x, y in zip(cal, coff)]); d2 = np.array([np.hypot(cal - x, coff - y).min() for x, y in zip(al[inn], off[inn])])
    tot[0] += len(cb); tot[1] += int((d1 <= 0.2).sum()); tot[2] += int(inn.sum()); tot[3] += int((d2 <= 0.2).sum()); first.append((i + 1, round(float(al[inn].min() - cal.min()), 2)))
print(f'[verify] causal bags at the last keyframe against the whole-pass census (graph poses): {tot[1]} of {tot[0]} inner-row census bags have a causal bag within 0.2 m; {tot[3]} of {tot[2]} causal inner-row bags have a census bag within 0.2 m')
print(f'[verify] first causal inner-row bag minus first census inner-row bag, along the lane (m), per lane: {first}')
# ---------------------------------------------------------------- 5. time alignment of the laser odometry
f = S['lo_file'].get('lo')
if f:
    lo = np.load(f); lt = lo['ts_ms'].astype(float); lp = lo['T'][:, :3, 3]; kt = np.array(EXP[a.rev]['ts_ms'], float); fin = np.isfinite(Cr).all((1, 2)); tg = np.arange(max(lt[0], kt[fin][0]) + 3000, min(lt[-1], kt[fin][-1]) - 3000, 100.0)
    def speed(ts, p, at): q = np.stack([np.interp(at + 500, ts, p[:, c]) - np.interp(at - 500, ts, p[:, c]) for c in range(3)], 1); return np.linalg.norm(q, axis=1)
    sg = speed(kt[fin], Cr[fin][:, :3, 3], tg); best = max(((np.corrcoef(sg, speed(lt, lp, tg + lag))[0, 1], lag) for lag in range(-2000, 2001, 50)))
    c0 = np.corrcoef(sg, speed(lt, lp, tg))[0, 1]; print(f'[verify] laser odometry clock: speed-profile correlation with the graph {c0:.4f} at zero lag; best {best[0]:.4f} at {best[1]:+d} ms (keyframe time = odometry time - lag)')
