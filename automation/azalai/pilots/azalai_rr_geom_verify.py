#!/usr/bin/env python3
"""azalai_rr_geom_verify.py (11 Nov 2026, A100 session) — independent recount of rr_geom<suffix>.json (azalai_rr_geom.py) from its
per-target records rr_geom_targets<suffix>.jsonl: evaluable counts by reason, the step-5 partner counts per datum and subset, the
exact 95 % intervals by this file's own binomial arithmetic (no scipy), per-lane counts. With --deep, step 5 is re-derived for every
evaluable pair by separately written arithmetic: the revisit census is carried into the REGISTRY lane frame (F_twin F_rev^-1, the
inverse direction of the script) and the nearest bag to the registry bag is taken in that frame's ground plane (lateral, along).
Exit status 1 on any disagreement. nerf_new python3.10:
  azalai_rr_geom_verify.py <dir> --suffix _apr1130_apr1400 [--deep]"""
import argparse, json, math, sys
from pathlib import Path
from collections import Counter
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--suffix', default=''); ap.add_argument('--deep', action='store_true'); a = ap.parse_args()
D = Path(a.dir); J = json.load(open(D / f'rr_geom{a.suffix}.json')); S = J['summary']; L = [json.loads(l) for l in open(D / f'rr_geom_targets{a.suffix}.jsonl')]; bad = []
def check(name, mine, theirs, tol=1e-9):
    ok = (abs(mine - theirs) <= tol) if isinstance(mine, float) and isinstance(theirs, (int, float)) else mine == theirs
    if not ok: bad.append(name); print(f'  DISAGREE {name}: recount {mine} vs summary {theirs}')
    return ok
def lpmf(k, n, p): return -math.inf if (p <= 0 and k > 0) or (p >= 1 and k < n) else math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + (k * math.log(p) if k else 0.0) + ((n - k) * math.log1p(-p) if n - k else 0.0)
def cdf(k, n, p): return sum(math.exp(lpmf(i, n, p)) for i in range(0, k + 1))
def clopper(k, n, al=0.05):
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - (1 - cdf(k - 1, n, p)), 1 - al / 2)
    high = 1.0 if k == n else solve(lambda p: cdf(k, n, p), al / 2)
    return low, high
ev = [r for r in L if 'skip' not in r]; DAT = S['datums']; fail = set(S['datum_failures'])
check('n_pairs', len(L), S['n_pairs']); check('evaluable', len(ev), S['evaluable']); check('not_evaluable', dict(Counter(r['skip'] for r in L if 'skip' in r)), S['not_evaluable'])
check('stored is_partner equals nearest == partner', sum(r['geometry'][dn]['is_partner'] != (r['geometry'][dn]['nearest'] == r['partner']) for r in ev for dn in DAT), 0)
sub = dict(all=ev, without_datum_failure_lanes=[r for r in ev if r['lane'] not in fail], datum_failure_lanes=[r for r in ev if r['lane'] in fail])
for dn in DAT:
    for name, R in sub.items():
        th = S['geometry_only'][dn][name]; k = sum(r['geometry'][dn]['nearest'] == r['partner'] for r in R); n = len(R); check(f'{dn} {name} n, partner', (n, k), (th['n'], th.get('partner')))
        if n:
            lo, hi = clopper(k, n); check(f'{dn} {name} interval low', lo, th['ci95_exact'][0], 1e-6); check(f'{dn} {name} interval high', hi, th['ci95_exact'][1], 1e-6); check(f'{dn} {name} line', lo > 0.80, th['lower_bound_above_0_80'])
            print(f"  {dn:9s} {name:28s} partner {k} of {n} = {k / n:.3f}, exact 95 % [{lo:.3f}, {hi:.3f}]")
for l, th in S['per_lane'].items():
    R = [r for r in ev if r['lane'] == int(l)]; check(f'per lane {l}', (len(R), {dn: sum(r['geometry'][dn]['nearest'] == r['partner'] for r in R) for dn in DAT}), (th['n'], th['partner']))
if a.deep:
    import numpy as np
    SK = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/sankofa/jirs_20261023')
    cd = lambda n: SK / ('camera_depth_feb_fixedreader' if n.startswith('feb') else 'lidar_fixedreader')
    Bt = np.load(cd(S['registry']) / f"census_{S['registry']}.npz")['bags_xyz'].astype(float); Br = np.load(cd(S['revisit']) / f"census_{S['revisit']}.npz")['bags_xyz'].astype(float)
    LF = {l['lane']: l for l in J['lanes_full']}; nd = 0
    G = None
    for r in ev:
        l = LF[r['lane']]; Ft = np.array(l['F_twin'])
        for dn in DAT:
            if dn == 'sitewide': continue                                     # the site-wide comparator is the truth construction itself; not re-derived here
            T = Ft @ np.linalg.inv(np.array(l['F_rev'][dn]))                   # revisit graph -> registry graph
            ok = np.isfinite(Br).all(1); idx = np.flatnonzero(ok); Pw = Br[idx] @ T[:3, :3].T + T[:3, 3]
            lat = (Pw - Ft[:3, 3]) @ Ft[:3, 0]; alo = (Pw - Ft[:3, 3]) @ Ft[:3, 2]; x0 = (Bt[r['twin_bag']] - Ft[:3, 3]) @ Ft[:3, 0]; a0 = (Bt[r['twin_bag']] - Ft[:3, 3]) @ Ft[:3, 2]
            j = int(np.argmin(np.hypot(lat - x0, alo - a0))); mine = int(idx[j]); nd += 1
            if mine != r['geometry'][dn]['nearest']: bad.append(f"deep {r['twin_bag']} {dn}"); print(f"  DISAGREE deep {r['twin_bag']} {dn}: nearest {mine} vs stored {r['geometry'][dn]['nearest']}")
    print(f'  step 5 re-derived in the registry frame for {nd} pair x datum cases')
print('ALL TABLES AGREE' if not bad else f'{len(bad)} DISAGREEMENTS'); sys.exit(1 if bad else 0)
