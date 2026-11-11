#!/usr/bin/env python3
"""azalai_h1_score_v1.py (11 Nov 2026) — score the H1 v1 records (azalai_h1_ground_v1.py) against the frozen gold with v0's rule,
unchanged: an instruction the gold marks unresolvable is not scored; a gold reject is correct iff the arm rejects; a count is
correct iff 'answer' is within 0.5 of the gold number; an 'any' gold is correct iff the arm names at least one target and all its
targets are in the gold set; otherwise the ordered target list must equal the gold list. A record with no final answer is wrong.
On the unresolvable instructions the arm's behaviour is tallied (rejected / answered / empty / no final).
Supplementary, outside the verdict: the ten co-reference instructions carry ids CO--01..10 in the bank and COR-01..10 in the gold
builder (azalai_h1_gold.py), so the frozen gold marks all ten 'no resolver'; they are scored here against the builder's COR
resolvers evaluated on registry_v0.json, in a separate block, without touching the gold file.

Usage:  python3 azalai_h1_score_v1.py <dir with registry_v0.json, instruction_bank_v0_gold.json> <records prefix> <summary.json>"""
import sys, json
from collections import defaultdict
d, pre, out = sys.argv[1:4]
G = json.load(open(f'{d}/instruction_bank_v0_gold.json')); R = json.load(open(f'{d}/registry_v0.json'))
ARMS = ('twin', 'enumerated'); NUMBERED = ['direct scene graph concepts', 'unambiguous goal command', 'co-reference resolution', 'spatial relation disambiguation', 'region-level disambiguation', 'ambiguous role assignment', 'failure']; NEW = ['identity', 'count', 'change']
TWIN_ONLY = ['IDE-01', 'IDE-02', 'IDE-03', 'IDE-10', 'COU-04', 'COU-06', 'COU-09', 'CHA-01', 'CHA-02', 'CHA-03', 'CHA-04', 'CHA-06', 'CHA-07', 'CHA-08', 'CHA-09']   # gold rule reads a twin-only field


def norm(t): return [str(x).strip() for x in (t if isinstance(t, list) else [t]) if x is not None and str(x).strip() != '']


def score(g, ans):   # v0's rule (azalai_h1_ground.py), verbatim
    if g.get('unresolvable'): return None
    if g['reject']: return bool(ans.get('reject'))
    if ans.get('reject'): return False
    gold = [str(x) for x in g['targets']]; got = norm(ans.get('targets', []))
    if g['category'] == 'count' and len(gold) == 1:
        v = ans.get('answer')
        try: return v is not None and abs(float(v) - float(gold[0])) < 0.5
        except Exception: return False
    if g['any']: return len(got) >= 1 and all(x in gold for x in got)
    return got == gold


def behaviour(rec):
    f = rec.get('final')
    if not f: return 'no final'
    if f.get('reject'): return 'rejected'
    return 'answered' if norm(f.get('targets', [])) or f.get('answer') is not None else 'empty'


# supplementary co-reference targets: the gold builder's COR resolvers, evaluated here
bags = [b for b in R['bags'] if isinstance(b['id'], int)]; byid = {b['id']: b for b in bags}
def row(r): return sorted([b for b in bags if b['row'] == r], key=lambda b: b['n_in_row'])
def nth(r, n): L = row(r); return [L[n - 1]['id']] if 0 < n <= len(L) else []
def near(r, i): b = byid[i]; return [min(row(r), key=lambda c: (c['x'] - b['x']) ** 2 + (c['z'] - b['z']) ** 2)['id']]
def nbr(i, s): b = byid[i]; return nth(b['row'], b['n_in_row'] + s)
def tallest(r): L = [b for b in row(r) if b['height_dec_m'] is not None]; return [max(L, key=lambda b: b['height_dec_m'])['id']]
COR = {'CO--01': [412] + nbr(412, 1), 'CO--02': nth(2, 30) + near(3, nth(2, 30)[0]), 'CO--03': nth(7, 1) + nth(7, 2), 'CO--04': [118] + nbr(118, 1) + [118], 'CO--05': tallest(5),
       'CO--07': [199, 200, 201], 'CO--08': nth(4, 2) + near(5, nth(4, 2)[0]), 'CO--10': [77] + near(8, 77)}

S = {arm: defaultdict(lambda: dict(n=0, correct=0, rejected=0, no_final=0, unresolvable=0, unres_rejected=0, unres_answered=0, unres_empty=0, unres_no_final=0)) for arm in ARMS}
per = {arm: {} for arm in ARMS}; cor = {arm: {} for arm in ARMS}; meta = {}
for arm in ARMS:
    recs = [json.loads(l) for l in open(f'{pre}_{arm}.jsonl') if l.strip()]; assert len(recs) == len(G) and {r['id'] for r in recs} == set(G), (arm, len(recs))
    meta[arm] = dict(model=recs[0]['model'], seed=recs[0]['seed'], options=recs[0]['options'], n=len(recs), tool_calls=sum(r['n_tool_calls'] for r in recs), seconds=round(sum(r['seconds'] for r in recs), 1),
                     status={s: sum(1 for r in recs if r['status'] == s) for s in sorted({r['status'] for r in recs})})
    for r in recs:
        g = G[r['id']]; c = S[arm][g['category']]; f = r.get('final') or {}; b = behaviour(r)
        if g.get('unresolvable'):
            c['unresolvable'] += 1; c['unres_' + b.replace(' ', '_')] += 1; per[arm][r['id']] = dict(correct=None, behaviour=b)
            if r['id'] in COR: cor[arm][r['id']] = bool(r.get('final')) and not f.get('reject') and norm(f.get('targets', [])) == [str(x) for x in COR[r['id']]]
            continue
        ok = bool(r.get('final')) and bool(score(g, f)); c['n'] += 1; c['correct'] += int(ok); c['rejected'] += int(b == 'rejected'); c['no_final'] += int(b == 'no final'); per[arm][r['id']] = dict(correct=ok, behaviour=b)
cats = [c for c in NUMBERED + NEW]
table = {c: {arm: dict(S[arm][c]) for arm in ARMS} for c in cats}
tot = {arm: {k: sum(S[arm][c][k] for c in cats) for k in ('n', 'correct', 'rejected', 'no_final', 'unresolvable', 'unres_rejected', 'unres_answered', 'unres_empty', 'unres_no_final')} for arm in ARMS}
# the decision rule (plan Section 5, H1), as written down in H1_v1_design.md before the scored pass
parity = {c: dict(n=S['twin'][c]['n'], twin=S['twin'][c]['correct'], enumerated=S['enumerated'][c]['correct'], diff=S['twin'][c]['correct'] - S['enumerated'][c]['correct']) for c in NUMBERED if S['twin'][c]['n'] > 0}
for v in parity.values(): v['within_1'] = abs(v['diff']) <= 1; v['not_more_than_1_below'] = v['diff'] >= -1
ext = {c: dict(n=S['twin'][c]['n'], twin=S['twin'][c]['correct'], enumerated=S['enumerated'][c]['correct'], twin_above_baseline=S['twin'][c]['correct'] > S['enumerated'][c]['correct']) for c in NEW}
to = {arm: dict(n=len(TWIN_ONLY), correct=sum(int(per[arm][i]['correct']) for i in TWIN_ONLY), rejected=sum(int(per[arm][i]['behaviour'] == 'rejected') for i in TWIN_ONLY)) for arm in ARMS}
verdict = dict(parity_two_sided=all(v['within_1'] for v in parity.values()), parity_one_sided=all(v['not_more_than_1_below'] for v in parity.values()), extension=all(v['twin_above_baseline'] for v in ext.values()))
verdict['H1_holds'] = verdict['parity_two_sided'] and verdict['extension']
summary = dict(run=meta, per_category=table, total=tot, numbered_parity=parity, new_categories=ext, twin_only_field_items=to, verdict=verdict,
               supplementary_coreference=dict(note='outside the verdict: gold builder COR resolvers; the frozen gold marks these unresolvable (id mismatch CO-- / COR-)', targets={k: v for k, v in COR.items()},
                                              **{arm: dict(n=len(COR), correct=sum(int(v) for v in cor[arm].values()), per_instruction=cor[arm]) for arm in ARMS}), per_instruction=per)
json.dump(summary, open(out, 'w'), indent=1)
print(f"{'category':34s} {'n':>3s} {'twin':>5s} {'enum':>5s} | rejects t/e | unresolvable n: rejected t/e, answered t/e")
for c in cats:
    t, e = S['twin'][c], S['enumerated'][c]; print(f"{c:34s} {t['n']:3d} {t['correct']:5d} {e['correct']:5d} | {t['rejected']}/{e['rejected']} | {t['unresolvable']}: {t['unres_rejected']}/{e['unres_rejected']}, {t['unres_answered']}/{e['unres_answered']}")
print('total', json.dumps(tot)); print('twin-only-field items', json.dumps(to)); print('supplementary co-reference', {arm: sum(int(v) for v in cor[arm].values()) for arm in ARMS}, 'of', len(COR)); print('verdict', json.dumps(verdict)); print('run', json.dumps(meta))
