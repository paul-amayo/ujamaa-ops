#!/usr/bin/env python3
"""azalai_rr_verify.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe": independent recount of every table of
rr_summary<suffix>.json from the per-target records alone (rr_targets<suffix>.jsonl). Every outcome is re-decided from the stored raw
quantities (M_t area, best IoU, the census name of m*, the partner; the nearest census bag of the geometry-only step; the nearest
anchor of a Part 1 landing) rather than read from the stored verdict; the exact interval and the sign tests are recomputed with
this file's own binomial arithmetic (no scipy); the Part 3 statistics are recomputed from the per-pass rows of rr_odo.json.
With --deep N (needs the box: prep.json, the renders, the masks) rule R is re-derived end to end for N seeded targets per datum
at e = 0 by separately written code: M_t from the full label map of the e = 0 render (not from the packed masks the scorer reads),
IoU by a plain loop over the decoded masks, the name by this file's own projection. Exit status 1 on any disagreement.
python3:  azalai_rr_verify.py <records dir> [--suffix _dec_apr1130] [--odo rr_odo.json] [--deep 40 --prep prep.json --renders dir]"""
import argparse, json, math, sys
from pathlib import Path
from collections import Counter
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--suffix', default=''); ap.add_argument('--odo', default=''); ap.add_argument('--deep', type=int, default=0); ap.add_argument('--prep', default=''); ap.add_argument('--renders', default=''); a = ap.parse_args()
D = Path(a.dir); S = json.load(open(D / f'rr_summary{a.suffix}.json')); L = [json.loads(l) for l in open(D / f'rr_targets{a.suffix}.jsonl')]; bad = []
def check(name, mine, theirs, tol=1e-9):
    ok = (abs(mine - theirs) <= tol) if isinstance(mine, float) and isinstance(theirs, (int, float)) else mine == theirs
    if not ok: bad.append(name); print(f'  DISAGREE {name}: recount {mine} vs summary {theirs}')
    return ok
# ---- binomial arithmetic
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
    low = 0.0 if k == 0 else solve(lambda p: 1 - (1 - cdf(k - 1, n, p)), 1 - al / 2)       # P(X >= k | p) = al / 2
    high = 1.0 if k == n else solve(lambda p: cdf(k, n, p), al / 2)                           # P(X <= k | p) = al / 2
    return low, high
def sign(g, b): return None if g + b == 0 else min(1.0, 2 * cdf(min(g, b), g + b, 0.5))
def decide(x, partner): return 'no pick' if (x['mt_px'] == 0 or x['iou'] <= 0 or x['mstar'] is None) else ('unnamed' if x['name'] is None else ('right plant' if x['name'] == partner else 'wrong plant'))
# ---- Part 2
P2 = [r for r in L if r['part'] == 2]; ev = [r for r in P2 if 'skip' not in r]; T = S['part2']; DAT = S['prep']['datums']; ES = S['prep']['sweep_m']; fail = set(S['prep']['datum_failures']); TW = [tw for tw in ('canon', 'bag') if tw in T['outcomes']['all'][DAT[0]]]
print(f"{S['twin']} -> {S['revisit']}: {len(P2)} pairs, {len(ev)} evaluable, not evaluable {dict(Counter(r['skip'] for r in P2 if 'skip' in r))}")
check('n_pairs', len(P2), T['n_pairs']); check('evaluable', len(ev), T['evaluable']); check('not_evaluable', dict(Counter(r['skip'] for r in P2 if 'skip' in r)), T['not_evaluable'])
stored = sum(decide(x, r['partner']) != x['outcome'] for r in ev for dn in DAT for tw in TW for x in r['R'][dn][tw]); check('stored outcomes that differ from the re-decided ones', stored, 0)
sub = dict(all=ev, without_datum_failure_lanes=[r for r in ev if r['lane'] not in fail], datum_failure_lanes=[r for r in ev if r['lane'] in fail], naming_confirmed=[r for r in ev if r['naming_vote'] == 'agree'], naming_not_confirmed=[r for r in ev if r['naming_vote'] != 'agree'],
           growth_stable=[r for r in ev if r['growth'] == 'stable'], growth_changed=[r for r in ev if r['growth'] in ('grown', 'shrunk')], growth_no_call=[r for r in ev if r['growth'] not in ('stable', 'grown', 'shrunk')])
def oc(r, dn, tw, n_e): return decide(r['R'][dn][tw][n_e], r['partner'])
for name, R in sub.items():
    for dn in DAT:
        for tw in TW:
            for n_e, e in enumerate(ES):
                assert all(r['R'][dn][tw][n_e]['e'] == e for r in R); c = Counter(oc(r, dn, tw, n_e) for r in R); th = T['outcomes'][name][dn][tw][f'{e:+.2f}']
                check(f'outcomes {name} {dn} {tw} e={e:+.2f}', (len(R), c['right plant'], c['wrong plant'], c['unnamed'], c['no pick']), (th['n'], th['right plant'], th['wrong plant'], th['unnamed'], th['no pick']))
for dn in DAT:
    for tw in TW:
        rate = {e: sum(oc(r, dn, tw, n_e) == 'right plant' for r in ev) / len(ev) for n_e, e in enumerate(ES)}; tol = max([x for x in (0.15, 0.30, 0.45, 0.90) if rate[0.0] > 0 and rate[x] >= 0.9 * rate[0.0] and rate[-x] >= 0.9 * rate[0.0]], default=0.0)
        check(f'tolerance {dn} {tw}', tol, T['tolerance'][dn][tw]['tolerance_m'])
    for name in ('all', 'without_datum_failure_lanes', 'datum_failure_lanes'):
        R = sub[name]; check(f'geometry only {dn} {name}', (len(R), sum(r['geometry'][dn]['nearest'] == r['partner'] for r in R)), (T['geometry_only'][dn][name]['n'], T['geometry_only'][dn][name]['partner']))
for l, th in T['per_lane'].items():
    R = [r for r in ev if r['lane'] == int(l)]; check(f'per lane {l}', (len(R), {dn: {tw: sum(oc(r, dn, tw, 0) == 'right plant' for r in R) for tw in TW} for dn in DAT}), (th['n'], th['right_plant_e0']))
def prim(R, dn='primary', tw='canon'): k = sum(oc(r, dn, tw, 0) == 'right plant' for r in R); return k, len(R), clopper(k, len(R))
for name, key in (('all', 'all_lanes'), ('without_datum_failure_lanes', 'without_datum_failure_lanes')):
    k, n, (lo, hi) = prim(sub[name]); th = T['primary_outcome'][key]; ok = check(f'primary outcome {key} count', (k, n), (th['right_plant'], th['n'])) & check(f'primary outcome {key} interval low', lo, th['ci95_exact'][0], 1e-6) & check(f'primary outcome {key} interval high', hi, th['ci95_exact'][1], 1e-6) & check(f'primary outcome {key} verdict', lo > 0.80, th['holds_on_logs'])
    print(f"  primary outcome ({key}): {k} of {n} = {k / n:.3f}, exact 95 % interval [{lo:.3f}, {hi:.3f}], holds on logs: {lo > 0.80}; summary agrees: {ok}")
    for kk, th in T['primary_outcome']['beside'].items():
        dn, tw = kk.rsplit('_', 1); k, n, (lo, hi) = prim(sub[name], dn, tw); check(f'beside {kk} {key}', (k, n, lo > 0.80), (th[key]['right_plant'], th[key]['n'], th[key]['holds_on_logs'])); check(f'beside {kk} {key} low', lo, th[key]['ci95_exact'][0], 1e-6)
for kk, th in T['paired_e0'].items():
    x, rest = kk.split('_vs_'); y, tw, name = rest.split('_', 2); R = sub[name]; g = sum(oc(r, x, tw, 0) == 'right plant' and oc(r, y, tw, 0) != 'right plant' for r in R); b = sum(oc(r, y, tw, 0) == 'right plant' and oc(r, x, tw, 0) != 'right plant' for r in R)
    p = sign(g, b); check(f'paired {kk}', (g, b), (th[f'right_only_{x}'], th[f'right_only_{y}'])); (p is None and th['sign_test_p'] is None) or check(f'paired {kk} p', float(p), th['sign_test_p'], 1e-9)
for dn in DAT:
    for tw in TW: c = [sum(oc(r, dn, tw, n_e) == 'right plant' for r in ev) for n_e in range(len(ES))]; print(f"  {dn:9s} {tw:5s} right plant of {len(ev)} at e = " + ', '.join(f'{e:+.2f}: {k}' for e, k in sorted(zip(ES, c))))
# ---- Part 1
if 'part1' in S:
    P1 = [r for r in L if r['part'] == 1]; check('part 1 targets', len(P1), S['part1']['n_targets'])
    for tw in TW:
        for name, R in (('all', P1), ('census_inner', [r for r in P1 if r['census_inner']])):
            lands = lambda r, tag: bool(r['poses'][tag][tw]['area_px'] > 0 and r['poses'][tag][tw].get('nearest_anchor') == r['twin_id'])
            for tag in ('C', 'L', 'R', 'H', 'T'):
                Q = [r for r in R if tag in r['poses'] and r['poses'][tag].get(tw) is not None]; th = S['part1']['poses'][f'{tw}_{name}_{tag}']
                check(f'part 1 {tw} {name} {tag}', (len(Q), sum(r['poses'][tag][tw]['area_px'] > 0 for r in Q), sum(lands(r, tag) for r in Q)), (th['n'], th['lit'], th['lands']))
            for kk in [k for k in S['part1']['paired'] if k.startswith(f'{tw}_{name}_')]:
                x, y = kk[len(f'{tw}_{name}_'):].split('_vs_'); B = [r for r in R if x in r['poses'] and y in r['poses'] and r['poses'][x].get(tw) is not None and r['poses'][y].get(tw) is not None]; th = S['part1']['paired'][kk]
                g = sum(lands(r, x) and not lands(r, y) for r in B); b = sum(lands(r, y) and not lands(r, x) for r in B); p = sign(g, b); check(f'part 1 paired {kk}', (len(B), g, b), (th['n'], th[f'only_{x}'], th[f'only_{y}'])); (p is None and th['sign_test_p'] is None) or check(f'part 1 paired {kk} p', float(p), th['sign_test_p'], 1e-9)
    print(f"  Part 1 recounted: {len(P1)} targets")
# ---- Part 3
if a.odo:
    O = json.load(open(a.odo)); main = O['summary']['main_surveys']; R = [r for n in main for r in O['surveys'][n].get('passes', []) if r['usable'] and r.get('plain')]
    def pct(x, q): x = sorted(x); h = (len(x) - 1) * q; i = int(math.floor(h)); return x[i] + (h - i) * (x[min(i + 1, len(x) - 1)] - x[i])
    for key in ('e_end_m', 'e_max_m'):
        x = [r['plain'][key] for r in R]; th = O['summary']['plain'][key]; check(f'odo {key} n', len(x), th['n']); check(f'odo {key} median', pct(x, 0.5), th['median'], 1e-9); check(f'odo {key} p90', pct(x, 0.9), th['p90'], 1e-9); check(f'odo {key} max', max(x), th['max'], 1e-9)
    print(f"  Part 3 recounted: {len(R)} passes, e_end median {pct([r['plain']['e_end_m'] for r in R], 0.5):.3f} m, p90 {pct([r['plain']['e_end_m'] for r in R], 0.9):.3f} m, max {max(r['plain']['e_end_m'] for r in R):.3f} m")
# ---- rule R end to end, by separately written code, for a seeded sample
if a.deep:
    import glob, random, numpy as np, pycocotools.mask as mu
    sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths; from rig_calib import load_rig
    P = json.load(open(a.prep)); K = load_rig(survey_paths.monos(P['rev_root']))['intrinsics']; C = np.array(json.load(open(P['stpg']))['experiences'][P['rev']]['c2w'], float); B = np.load(P['census_rev'])['bags_xyz'].astype(float); ent = {}
    for f in sorted(glob.glob(P['rev_root'] + '/prod/bateleur/sam3_growbag/clip_*/frame_entries.json')):
        c = json.load(open(f))
        for li, fr in enumerate(c['frames']): ent.setdefault(int(fr['kf_idx']), c['frame_entries'].get(str(li), []))
    random.seed(0); pick = random.sample(ev, min(a.deep, len(ev))); nd = 0
    for r in pick:
        for dn in DAT:
            for tw in TW:
                x = r['R'][dn][tw][0]; z = np.load(Path(a.renders) / f"p2_{r['twin_bag']}_{dn}_0.npz")
                if tw not in z.files: continue
                Mt = z[tw] == r['twin_id']; best, bi = 0.0, None
                for i, en in enumerate(ent.get(r['R'][dn]['stop_k'], [])):
                    rr = dict(en['rle']); rr['counts'] = rr['counts'].encode(); m = mu.decode(rr).astype(bool); mh = m[::2, ::2]; u_ = np.logical_or(mh, Mt).sum(); v_ = np.logical_and(mh, Mt).sum() / u_ if u_ else 0.0
                    if v_ > best: best, bi, mfull = float(v_), i, m
                if bi is None: mine = 'no pick'; nm = None
                else:
                    Tcw = np.linalg.inv(C[r['R'][dn]['stop_k']]); cands = []
                    for j in range(len(B)):
                        if not np.isfinite(B[j]).all(): continue
                        pc = Tcw[:3, :3] @ B[j] + Tcw[:3, 3]
                        if pc[2] <= 0: continue
                        uu = K['fx'] * pc[0] / pc[2] + K['cx']; vv = K['fy'] * pc[1] / pc[2] + K['cy']
                        if 0 <= uu < 1280 and 0 <= vv < 720 and mfull[int(vv), int(uu)]: cands.append((float(np.linalg.norm(pc)), j))
                    nm = min(cands)[1] if cands else None; mine = 'unnamed' if nm is None else ('right plant' if nm == r['partner'] else 'wrong plant')
                ok = mine == x['outcome'] and int(Mt.sum()) == x['mt_px'] and abs(best - x['iou']) < 1e-9 and nm == x['name']; nd += 1
                if not ok: bad.append(f"deep {r['twin_bag']} {dn} {tw}"); print(f"  DISAGREE deep {r['twin_bag']} {dn} {tw}: mine {mine} iou {best} name {nm} px {int(Mt.sum())} vs stored {x['outcome']} {x['iou']} {x['name']} {x['mt_px']}")
    print(f'  rule R re-derived end to end for {len(pick)} seeded targets ({nd} target x datum x twin cases at e = 0)')
print('ALL TABLES AGREE' if not bad else f'{len(bad)} DISAGREEMENTS'); sys.exit(1 if bad else 0)
