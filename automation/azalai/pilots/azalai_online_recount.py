#!/usr/bin/env python3
"""azalai_online_recount.py (A100 agent, 11 Nov 2026): AZ2 online pilot, a second independent recount, written by the agent
that took the test over. It shares no code with azalai_online_score.py or azalai_online_verify.py.

For one run (registry, revisit[, direct]) it recomputes from the per-target records (online_targets_*.jsonl), the per-keyframe
curves (online_curves_*.json) and, for the truth, the census and placement files themselves:
  1. the truth pairs: mutual nearest neighbours within 0.5 m in the December ground plane (x, z), each pass carried by its own
     placement G_global (December = identity), finite bags only; compared with the records' (registry_bag, partner) pairs;
  2. the target bookkeeping (reasons, targets, stop-evaluable);
  3. every variant's right / neighbour / other / no-stop counts, the exact (Clopper-Pearson) 95 % interval by bisection on the
     binomial distribution with exact integer arithmetic, and the paired exact two-sided sign tests;
  4. the counting baseline overall and by n band, with the median |count offset|;
  5. the station-error summaries (|e| median, 90th percentile, maximum, share above S/2, 5 m bands, datum failures, first and
     last error per lane, the distance after which |e| stays below S/2) from the curves.
Everything is compared with online_summary_*.json; every disagreement is printed. Writes recount_<tag>.json (the numbers the
results file quotes). Python: nerf_new pixi python3.10.
  azalai_online_recount.py --reg dec --rev apr1130 --records <dir> --out <dir> [--direct]"""
import argparse, json, math
from fractions import Fraction
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reg', required=True); ap.add_argument('--rev', required=True); ap.add_argument('--records', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--direct', action='store_true')
args = ap.parse_args()
tag = f'{args.reg}_{args.rev}' + ('_direct' if args.direct else '')
rd = Path(args.records)
summary = json.load(open(rd / f'online_summary_{tag}.json'))
records = [json.loads(line) for line in open(rd / f'online_targets_{tag}.jsonl')]
curves = json.load(open(rd / f'online_curves_{tag}.json'))
CENSUS = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/sankofa/jirs_20261023/lidar_fixedreader')
problems = []


def same(name, mine, theirs, tol=1e-9, atol=0.0):
    """equal; floats within tol (relative) or atol (absolute: the curves store errors and distances rounded to 0.1 mm)"""
    if isinstance(mine, str) or isinstance(theirs, str) or mine is None or theirs is None:
        ok = mine == theirs
    elif isinstance(mine, (float, np.floating)) or isinstance(theirs, float):
        ok = abs(float(mine) - float(theirs)) <= max(tol * max(1.0, abs(float(theirs))), atol)
    else:
        ok = mine == theirs
    if not ok:
        problems.append((name, mine, theirs))


# ------------------------------------------------------------------ exact binomial tools (integer arithmetic)
def binom_pmf_exact(n):
    return [Fraction(math.comb(n, i), 2 ** n) for i in range(n + 1)]


def cdf_le(k, n, p):   # P(X <= k), X ~ Bin(n, p), float p
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n):
    """exact 95 % interval: lower = p with P(X >= k | p) = 0.025, upper = p with P(X <= k | p) = 0.025 (bisection)"""
    if n == 0:
        return [None, None]
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < target: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)
    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - cdf_le(k - 1, n, p), 0.025)   # increasing in p
    upper = 1.0 if k == n else 1.0 - solve(lambda q: 1.0 - cdf_le(n - k - 1, n, q), 0.025)  # by symmetry on q = 1 - p
    return [lower, upper]


def sign_test_p(x, y):
    n = x + y
    if n == 0:
        return 1.0
    pm = binom_pmf_exact(n); m = min(x, y)
    tail = sum(pm[: m + 1])
    return float(min(Fraction(1), 2 * tail)) if x != y else 1.0


# ------------------------------------------------------------------ 1. truth pairs from the census and placement files
def placed(name):
    return np.eye(4) if name == 'dec' else np.load(CENSUS / f'placed_{name}.npz')['G_global'].astype(float)


def bags(name):
    return np.load(CENSUS / f'census_{name}.npz')['bags_xyz'].astype(float)


truth_note = 'not re-derived (direct-fit sensitivity)'
rec_pairs = sorted((r['registry_bag'], r['partner']) for r in records)
if not args.direct:
    Bt, Br = bags(args.reg), bags(args.rev)
    it = np.flatnonzero(np.isfinite(Bt).all(1)); ir = np.flatnonzero(np.isfinite(Br).all(1))
    Gt, Gr = placed(args.reg), placed(args.rev)
    Pt = (Bt[it] @ Gt[:3, :3].T + Gt[:3, 3])[:, [0, 2]]; Pr = (Br[ir] @ Gr[:3, :3].T + Gr[:3, 3])[:, [0, 2]]
    D2 = ((Pt[:, None, :] - Pr[None, :, :]) ** 2).sum(-1)           # brute force, no KD-tree
    a_best = D2.argmin(1); b_best = D2.argmin(0)
    mine = sorted((int(it[i]), int(ir[a_best[i]])) for i in range(len(it)) if b_best[a_best[i]] == i and D2[i, a_best[i]] <= 0.25)
    same('truth.mutual_nn_count', len(mine), summary['truth']['mutual_nn'])
    if summary['truth']['source'] == 'pilot record':
        extra = set(rec_pairs) - set(mine)
        same('truth.pilot_pairs_not_mutual', len(extra), 0)
        truth_note = f'{len(rec_pairs)} pilot pairs, all among the {len(mine)} mutual nearest neighbours re-derived here' if not extra else f'{len(extra)} pilot pairs NOT mutual'
    else:
        same('truth.pairs_identical', rec_pairs == mine, True)
        truth_note = f'{len(mine)} mutual nearest neighbours re-derived here; ' + ('identical to the records' if rec_pairs == mine else 'DIFFERENT from the records')
    same('truth.readable_by_rule', max(float(np.load(CENSUS / f'placed_{n}.npz')['global_fit_resid_med']) if n != 'dec' else 0.0 for n in (args.reg, args.rev)) <= 0.15, summary['truth']['readable_by_rule'])

# ------------------------------------------------------------------ 2. bookkeeping
targets = [r for r in records if 'skip' not in r]
stop_ok = [r for r in targets if r['stop_evaluable']]
reasons = {}
for r in records:
    if 'skip' in r: reasons[r['skip']] = reasons.get(r['skip'], 0) + 1
same('n_pairs', len(records), summary['n_pairs']); same('targets', len(targets), summary['targets']); same('stop_evaluable', len(stop_ok), summary['stop_evaluable'])
same('not_target', reasons, summary['not_target'])
ns_reasons = {}
for r in targets:
    if not r['stop_evaluable']: ns_reasons[r['stop_skip']] = ns_reasons.get(r['stop_skip'], 0) + 1
same('not_stop_evaluable', ns_reasons, summary['not_stop_evaluable'])


# ------------------------------------------------------------------ 3. variants and paired tests
def tally(outs):
    n = len(outs); k = sum(o == 'right' for o in outs)
    return dict(n=n, right=k, neighbour=sum(o == 'neighbour' for o in outs), other=sum(o == 'other' for o in outs), rate=(k / n if n else None), ci95=clopper_pearson(k, n))


out = dict(tag=tag, truth=truth_note, n_pairs=len(records), targets=len(targets), stop_evaluable=len(stop_ok), not_target=reasons, not_stop_evaluable=ns_reasons,
           standoff_d_m=summary['standoff_d_m'], spacing_m=summary['spacing_m'], half_spacing_m=summary['half_spacing_m'], variants={}, paired={}, counting={}, lanes={})
for v, want in summary['variants'].items():
    rows = [r for r in stop_ok if 'outcome' in r['var'][v]]
    for r in rows:   # right iff the arrival is the partner
        o = r['var'][v]
        if o['no_stop']: same(f'{v}.no_stop_is_other.{r["registry_bag"]}', o['outcome'], 'other')
        else: same(f'{v}.right_iff_partner.{r["registry_bag"]}', o['arrival'] == r['partner'], o['outcome'] == 'right')
    t = tally([r['var'][v]['outcome'] for r in rows]); t['no_stop'] = sum(r['var'][v]['no_stop'] for r in rows); t['skipped_no_lo'] = sum('outcome' not in r['var'][v] for r in stop_ok)
    for key in ('n', 'right', 'neighbour', 'other', 'no_stop', 'skipped_no_lo'): same(f'{v}.{key}', t[key], want[key])
    if t['n']:
        same(f'{v}.rate', t['rate'], want['rate']); same(f'{v}.ci_lo', t['ci95'][0], want['ci95'][0], 1e-7); same(f'{v}.ci_hi', t['ci95'][1], want['ci95'][1], 1e-7)
    t['by_lane'] = {}
    for ln in sorted({r['lane'] for r in rows}):
        tl = tally([r['var'][v]['outcome'] for r in rows if r['lane'] == ln]); t['by_lane'][str(ln)] = {k: tl[k] for k in ('n', 'right', 'neighbour', 'other')}
        for key in ('n', 'right', 'neighbour', 'other'): same(f'{v}.lane{ln}.{key}', tl[key], want['by_lane'][str(ln)][key])
    out['variants'][v] = t


def outcome_of(r, name):
    return r['counting']['outcome'] if name == 'counting' else r['var'][name]['outcome']


for key, want in summary['paired'].items():
    x, y = key.split('_vs_')
    rows = [r for r in stop_ok if all(n == 'counting' or 'outcome' in r['var'][n] for n in (x, y))]
    fo = sum(outcome_of(r, x) == 'right' and outcome_of(r, y) != 'right' for r in rows); so = sum(outcome_of(r, x) != 'right' and outcome_of(r, y) == 'right' for r in rows)
    p = sign_test_p(fo, so)
    same(f'paired.{key}.n', len(rows), want['n']); same(f'paired.{key}.first_only', fo, want['first_only']); same(f'paired.{key}.second_only', so, want['second_only'])
    if not (abs(p - want['p']) <= 1e-9 * max(1e-300, want['p']) or abs(math.log10(max(p, 1e-300)) - math.log10(max(want['p'], 1e-300))) < 1e-6): problems.append((f'paired.{key}.p', p, want['p']))
    out['paired'][key] = dict(n=len(rows), first_only=fo, second_only=so, p=p)

# ------------------------------------------------------------------ 4. counting
for label, rows, want in (('all_targets', targets, summary['counting']['all_targets']), ('stop_evaluable', stop_ok, summary['counting']['stop_evaluable'])):
    t = tally([r['counting']['outcome'] for r in rows]); out['counting'][label] = t
    for key in ('n', 'right', 'neighbour', 'other'): same(f'counting.{label}.{key}', t[key], want[key])
    if t['n']: same(f'counting.{label}.ci_lo', t['ci95'][0], want['ci95'][0], 1e-7); same(f'counting.{label}.ci_hi', t['ci95'][1], want['ci95'][1], 1e-7)
same('counting.no_nth_bag', sum(r['counting']['no_nth_bag'] for r in targets), summary['counting']['no_nth_bag'])
out['counting']['by_n'] = {}
for lo, hi, label in ((1, 10, '1-10'), (11, 20, '11-20'), (21, 30, '21-30'), (31, 40, '31-40'), (41, 10 ** 6, '41-up')):
    rows = [r for r in targets if lo <= r['n'] <= hi]; t = tally([r['counting']['outcome'] for r in rows])
    t['abs_offset_med'] = float(np.median([abs(r['counting']['count_offset']) for r in rows])) if rows else None
    want = summary['counting']['by_n'][label]
    for key in ('n', 'right', 'neighbour', 'other'): same(f'counting.by_n.{label}.{key}', t[key], want[key])
    if rows: same(f'counting.by_n.{label}.abs_offset_med', t['abs_offset_med'], want['abs_offset_med'])
    out['counting']['by_n'][label] = {k: t[k] for k in ('n', 'right', 'neighbour', 'other', 'rate', 'abs_offset_med')}

# ------------------------------------------------------------------ 5. station error from the curves
half = curves['half_spacing_m']; same('half_spacing', half, summary['half_spacing_m'])
lanes_sum = {l['lane']: l for l in summary['lanes']}
for v, want in summary['variants'].items():
    errs, Ls = [], []; fails = []; per_lane = {}
    for lane in curves['lanes']:
        c = lane['var'].get(v)
        if c is None: continue
        e = np.array([np.nan if x is None else x for x in c['err']], float); L = np.array(lane['L'], float); ok = np.isfinite(e)
        if not ok.any(): continue
        errs.append(e[ok]); Ls.append(L[ok]); j0 = int(np.flatnonzero(ok)[0]); first = float(e[j0]); last = float(e[ok][-1])
        over = np.flatnonzero(ok & (np.abs(e) >= half))
        if len(over) and over[-1] == len(e) - 1: settle = 'never'
        elif not len(over): settle = float(L[j0])
        else: settle = float(L[over[-1] + 1])
        if abs(first) > half: fails.append(lane['lane'])
        per_lane[str(lane['lane'])] = dict(first_belief_at_m=float(L[j0]), err_first_m=first, err_end_m=last, abs_max_m=float(np.abs(e[ok]).max()), stays_below_half_after_m=settle, datum_failure=abs(first) > half)
        w = lanes_sum[lane['lane']]['var'][v]
        for key in ('first_belief_at_m', 'err_first_m', 'err_end_m', 'abs_max_m'): same(f'{v}.lane{lane["lane"]}.{key}', per_lane[str(lane['lane'])][key], w[key], atol=1e-4)
        same(f'{v}.lane{lane["lane"]}.stays_below_half_after_m', settle, w['stays_below_half_after_m'], atol=1e-4)
        same(f'{v}.lane{lane["lane"]}.datum_failure', abs(first) > half, w['datum_failure'])
    if not errs: continue
    E = np.abs(np.concatenate(errs)); LL = np.concatenate(Ls); se = want['station_error']
    mine = dict(n_keyframes=int(len(E)), abs_med_m=float(np.median(E)), abs_p90_m=float(np.percentile(E, 90)), abs_max_m=float(E.max()), share_over_half=float((E > half).mean()), datum_failures=fails)
    for key in ('n_keyframes', 'abs_med_m', 'abs_p90_m', 'abs_max_m', 'share_over_half', 'datum_failures'): same(f'{v}.station.{key}', mine[key], se[key], atol=1e-4)
    mine['bands_5m'] = {}
    for b in range(0, 50, 5):
        m = (LL >= b) & (LL < b + 5); label = f'{b}-{b + 5}'
        if not m.any(): same(f'{v}.band{label}', None, se['bands_5m'][label]); continue
        mb = dict(n=int(m.sum()), abs_med_m=float(np.median(E[m])), abs_p90_m=float(np.percentile(E[m], 90)), share_over_half=float((E[m] > half).mean()))
        for key in mb: same(f'{v}.band{label}.{key}', mb[key], se['bands_5m'][label][key], atol=1e-4)
        mine['bands_5m'][label] = mb
    out['variants'][v]['station_error'] = mine; out['lanes'][v] = per_lane

Path(args.out).mkdir(parents=True, exist_ok=True)
json.dump(dict(**out, problems=[list(map(str, p)) for p in problems]), open(Path(args.out) / f'recount_{tag}.json', 'w'), indent=1)
print(f'[recount] {tag}: {len(records)} pairs, {len(targets)} targets, {len(stop_ok)} stop-evaluable; truth: {truth_note}')
for v in ('M', 'a', 'b', 'cp', 'cm', 'S', 'O'):
    if v in out['variants'] and out['variants'][v]['n']:
        t = out['variants'][v]; print(f"  {v:3s} {t['right']}/{t['n']} = {t['rate']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]  datum failures {t.get('station_error', {}).get('datum_failures')}")
c = out['counting']['stop_evaluable']; print(f"  counting {c['right']}/{c['n']} = {c['rate']:.3f}")
print(f'[recount] {tag}: ' + ('ALL AGREE with the summary' if not problems else f'{len(problems)} DISAGREEMENTS'))
for p in problems[:30]: print('   ', p)
