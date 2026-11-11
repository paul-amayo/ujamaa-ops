#!/usr/bin/env python3
"""azalai_h3_twin_score.py (11 Nov 2026) — RSS Azalai Stage 1 pilot H3 with the twin render, step 3 of 3: the arrival check, as fixed
in Azalai_RSS/numbers/pilots_20261111/H3_twin_design.md BEFORE this script was first run. For every pair of prep.json and both
seasons (April 11:39 = the revisit; December held-out keyframes = the same-season comparator), with the live frame of the correct
arrival (the goal keyframe) and of the wrong arrival (the neighbour's goal keyframe), and the twin always rendered at the BELIEVED
pose (the target's goal):
  rule A (canon, bag)  m* = the live SAM3 mask with the highest IoU against the target's rendered mask; arrived if IoU(m*, target)
                       > 0 and > IoU(m*, neighbour) for each row neighbour, all masks from the one believed-pose render;
  rule B (bag)         each candidate's mask = the largest connected component of its rendered mask in its own view (target: the
                       believed pose; neighbour: the believed pose translated by the anchor offset), translated to the centroid of
                       m*; arrived if the target's IoU with m* is strictly the highest (and > 0).
No threshold, no tolerance. Writes h3_twin_pairs.jsonl (one line per pair and season, unevaluable pairs included with their reason)
and h3_twin_summary.json (confusion matrices: all / unchanged / changed / no call / wrong bag is a candidate / naming confirmed; the
positional stand-in of 29 Oct on the same pairs; sign tests of the two readings). nerf_new python3.10:
  azalai_h3_twin_score.py <prep.json> <renders dir> <apr survey root> <dec survey root> <out dir>"""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import Counter
from scipy import ndimage
from scipy.stats import binomtest
import pycocotools.mask as mu
prep, RD, AR, DR, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]); OUT.mkdir(parents=True, exist_ok=True)
P = json.load(open(prep)); UNLAB = 65535
def load_entries(root):
    ents = {}
    for f in sorted(glob.glob(str(root / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
        c = json.load(open(f))
        for li, fr in enumerate(c['frames']):
            k = int(fr['kf_idx'])
            if k not in ents: ents[k] = [x['rle'] for x in c['frame_entries'].get(str(li), [])]
    return ents
ENT = dict(apr=load_entries(AR), dec=load_entries(DR)); cache = {}
def live(season, k):
    """the live frame's SAM3 grow-bag masks at the render scale (every second pixel)"""
    if (season, k) not in cache:
        ms = []
        for rle in ENT[season].get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool)[::2, ::2])
        if len(cache) > 400: cache.clear()
        cache[(season, k)] = ms
    return cache[(season, k)]
def iou(x, y):
    u = (x | y).sum(); return float((x & y).sum() / u) if u else 0.0
def largest(m):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    return m if n <= 1 else lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
def centroid(m): v, u = np.nonzero(m); return float(u.mean()), float(v.mean())
def moved(m, du, dv):
    o = np.zeros_like(m); h, w = m.shape; v, u = np.nonzero(m); u = u + du; v = v + dv; ok = (u >= 0) & (u < w) & (v >= 0) & (v < h); o[v[ok], u[ok]] = True; return o
def rule_a(ids, cand, ms):
    Rt = ids == cand['t']
    if not Rt.any(): return dict(arrived=False, reason='target not rendered', iou_t=0.0, iou_n=0.0, n_masks=len(ms)), None
    if not ms: return dict(arrived=False, reason='no live masks', iou_t=0.0, iou_n=0.0, n_masks=0), None
    i = int(np.argmax([iou(m, Rt) for m in ms])); m = ms[i]; it = iou(m, Rt); nb = {c: iou(m, ids == cand[c]) for c in ('prev', 'next') if cand[c] is not None}; i_n = max(nb.values()) if nb else 0.0
    return dict(arrived=bool(it > 0 and it > i_n), iou_t=it, iou_n=i_n, iou_prev=nb.get('prev'), iou_next=nb.get('next'), n_masks=len(ms), target_px=int(Rt.sum()), live_px=int(m.sum())), (m if it > 0 else None)
def rule_b(views, cand, m):
    if m is None: return dict(arrived=False, reason='no live mask under test')
    cu, cv = centroid(m); sc = {}
    for c in ('t', 'prev', 'next'):
        if cand[c] is None: continue
        S = views.get(c); S = None if S is None else (S == cand[c])
        if S is None or not S.any(): sc[c] = 0.0; continue
        S = largest(S); su, sv = centroid(S); sc[c] = iou(m, moved(S, int(round(cu - su)), int(round(cv - sv))))
        if c == 't': off = (cu - su, cv - sv)
    nbs = [sc[c] for c in ('prev', 'next') if c in sc]; r = dict(arrived=bool(sc['t'] > 0 and all(sc['t'] > x for x in nbs)), s_t=sc['t'], s_prev=sc.get('prev'), s_next=sc.get('next'))
    if sc['t'] > 0: r['offset_px'] = [float(off[0]), float(off[1])]   # live mask centroid minus the target's rendered bag mask, at the render scale
    return r
def npz(job):
    f = RD / f'{job}.npz'; return np.load(f) if f.exists() else None
lines = []
for season, L in (('apr', P['pairs']), ('dec', P['same_season'])):
    for p in L:
        r = dict(season=season, dec=p['dec'], growth=p.get('growth'), naming_vote=p.get('naming_vote'))
        if season == 'apr': r['positional'] = p['positional']
        if 'skip' in p: r['skip'] = p['skip']; lines.append(r); continue
        t = p['dec']; z0 = npz(f'{season}_{t}_V0')
        if z0 is None: r['skip'] = 'no render'; lines.append(r); continue
        cand = dict(p['twin']); zs = dict(t=z0, prev=npz(f'{season}_{t}_Vprev') if cand['prev'] is not None else None, next=npz(f'{season}_{t}_Vnext') if cand['next'] is not None else None)
        r.update(lane=p['lane'], goal_k=p['goal_k'], wrong_k=p['wrong_k'], twin=cand, n1_is_candidate=p['n1_is_candidate'], target_range_m=p['target_range_m'], true_wrong_displacement_m=p['true_wrong_displacement_m'])
        for case, k in (('correct', p['goal_k']), ('wrong', p['wrong_k'])):
            ms = live(season, k); o = {}
            o['A_canon'], _ = rule_a(z0['canon'], cand, ms); o['A_bag'], m = rule_a(z0['bag'], cand, ms)
            o['B_bag'] = rule_b({c: (z['bag'] if z is not None else None) for c, z in zs.items()}, cand, m)
            r[case] = o
        lines.append(r)
        if len(lines) % 100 == 0: print('[h3-twin]', len(lines), flush=True)
with open(OUT / 'h3_twin_pairs.jsonl', 'w') as f:
    for r in lines: f.write(json.dumps(r) + '\n')
# ---------------------------------------------------------------- confusion matrices and the two readings
def sign(g, b): return float(binomtest(g, g + b).pvalue) if g + b else None
def confusion(R, get):
    """R: evaluated records; get(r, case) -> arrived"""
    c = [bool(get(r, 'correct')) for r in R]; w = [bool(get(r, 'wrong')) for r in R]; n = len(R)
    good = sum(a and not b for a, b in zip(c, w)); bad = sum((not a) and b for a, b in zip(c, w))
    return dict(n=n, correct_confirmed=sum(c), correct_rejected=n - sum(c), wrong_confirmed=sum(w), wrong_caught=n - sum(w), separation=dict(good=good, bad=bad, sign_test_p=sign(good, bad)))
RULES = ('A_canon', 'A_bag', 'B_bag'); S = dict(design='Azalai_RSS/numbers/pilots_20261111/H3_twin_design.md', prep=P['summary'])
for season in ('apr', 'dec'):
    A = [r for r in lines if r['season'] == season]; ev = [r for r in A if 'skip' not in r]
    subsets = dict(all=ev, unchanged=[r for r in ev if r['growth'] == 'stable'], changed=[r for r in ev if r['growth'] in ('grown', 'shrunk')], no_call=[r for r in ev if r['growth'] not in ('stable', 'grown', 'shrunk')],
                   wrong_bag_is_candidate=[r for r in ev if r['n1_is_candidate']], naming_confirmed=[r for r in ev if r['naming_vote'] == 'agree'],
                   candidate_and_naming=[r for r in ev if r['n1_is_candidate'] and r['naming_vote'] == 'agree'])
    D = dict(n_pairs=len(A), evaluable=len(ev), not_evaluable=dict(Counter(r['skip'] for r in A if 'skip' in r)))
    for name, R in subsets.items():
        d = {rule: confusion(R, lambda r, case, rule=rule: r[case][rule]['arrived']) for rule in RULES}
        for rule in RULES: d[rule]['reasons'] = dict(Counter(f"{case}: {r[case][rule]['reason']}" for r in R for case in ('correct', 'wrong') if 'reason' in r[case][rule]))
        if season == 'apr':
            d['positional_29oct'] = confusion(R, lambda r, case: r['positional'][case])
            for rule in RULES:   # reading 2: against the positional stand-in on the same pairs
                cw = sum((not r['wrong'][rule]['arrived']) and r['positional']['wrong'] for r in R); pw = sum(r['wrong'][rule]['arrived'] and not r['positional']['wrong'] for r in R)
                cc = sum(r['correct'][rule]['arrived'] and not r['positional']['correct'] for r in R); pc = sum((not r['correct'][rule]['arrived']) and r['positional']['correct'] for r in R)
                d[rule]['vs_positional'] = dict(wrong_caught_only_by_twin=cw, wrong_caught_only_by_positional=pw, sign_test_p=sign(cw, pw), correct_confirmed_only_by_twin=cc, correct_confirmed_only_by_positional=pc, correct_sign_test_p=sign(cc, pc))
        D[name] = d
    if season == 'apr':
        D['positional_29oct_all_552'] = confusion(A, lambda r, case: r['positional'][case])
        D['positional_29oct_not_evaluable'] = confusion([r for r in A if 'skip' in r], lambda r, case: r['positional'][case])
    med = lambda xs: float(np.median(xs)) if len(xs) else None
    D['iou'] = {rule: dict(correct_target_median=med([r['correct'][rule]['iou_t'] for r in ev]), correct_neighbour_median=med([r['correct'][rule]['iou_n'] for r in ev]),
                           wrong_target_median=med([r['wrong'][rule]['iou_t'] for r in ev]), wrong_neighbour_median=med([r['wrong'][rule]['iou_n'] for r in ev])) for rule in ('A_canon', 'A_bag')}
    D['iou']['B_bag'] = dict(correct_target_median=med([r['correct']['B_bag']['s_t'] for r in ev if 's_t' in r['correct']['B_bag']]), wrong_target_median=med([r['wrong']['B_bag']['s_t'] for r in ev if 's_t' in r['wrong']['B_bag']]))
    off = np.array([r['correct']['B_bag']['offset_px'] for r in ev if 'offset_px' in r['correct']['B_bag']])
    if len(off): D['pose_transfer_offset_px_render_scale'] = dict(n=len(off), median_abs_du=float(np.median(np.abs(off[:, 0]))), median_abs_dv=float(np.median(np.abs(off[:, 1]))), median_norm=float(np.median(np.linalg.norm(off, axis=1))),
                                                                  p90_norm=float(np.percentile(np.linalg.norm(off, axis=1), 90)), median_du=float(np.median(off[:, 0])), median_dv=float(np.median(off[:, 1])),
                                                                  note='live mask under test (centroid) minus the largest component of the target\'s rendered bag mask, correct arrivals, pixels at 640 x 360; x2 for the full image')
    S['april_1130' if season == 'apr' else 'december_held_out'] = D
json.dump(S, open(OUT / 'h3_twin_summary.json', 'w'), indent=1)
for season in ('april_1130', 'december_held_out'):
    D = S[season]; print(f"== {season}: {D['evaluable']} of {D['n_pairs']} evaluable; not evaluable {D['not_evaluable']}")
    for name in ('all', 'unchanged', 'changed', 'no_call', 'wrong_bag_is_candidate', 'naming_confirmed', 'candidate_and_naming'):
        for rule in RULES + (('positional_29oct',) if season == 'april_1130' else ()):
            c = D[name][rule]; print(f"   {name:24s} {rule:16s} n {c['n']:4d} | correct confirmed {c['correct_confirmed']:4d} rejected {c['correct_rejected']:4d} | wrong confirmed {c['wrong_confirmed']:4d} caught {c['wrong_caught']:4d} | good {c['separation']['good']} bad {c['separation']['bad']} p {c['separation']['sign_test_p']}")
