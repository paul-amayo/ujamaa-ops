#!/usr/bin/env python3
"""azalai_h1p_rerun_outcomes.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3): for the determinism re-run (change
category, both sets, both arms) and for the post hoc warm-start diagnostic, where each differing trace first diverges and whether
the outcome under the frozen gold differs. Descriptive; uses the recount's own scorer and replay.
usage: python3 azalai_h1p_rerun_outcomes.py <dir> <out.json>"""
import sys, os, json, importlib.util
D, OUT = sys.argv[1:3]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'azalai_h1p_recount.py')).read()
head = src.split('# ---------------------------------------------------------------- load and replay every run')[0].replace('SUM = json.load(open(SUMF)); FAIL = json.load(open(FAILF))', '')
sys.argv = ['x', D, 'none', 'none', 'none']; g = {'__name__': 'recount_defs', '__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'azalai_h1p_recount.py')}
exec(compile(head, 'recount_defs', 'exec'), g)
def load(p): return {r['id']: r for r in map(json.loads, open(f'{D}/{p}.jsonl'))}
def diverge(r1, r2):
    for k, (a, b) in enumerate(zip(r1['steps'], r2['steps'])):
        if a.get('raw') != b.get('raw'): return k
    return min(len(r1['steps']), len(r2['steps']))
res = {}; pairs = [('A', 'h1p_setA', 'h1p_setA_change_rerun', 'A'), ('B', 'h1p_setB', 'h1p_setB_change_rerun', 'B')]
for name, first, rerun, s in pairs:
    for a in g['ARMS']:
        f1, f2 = load(f'{first}_{a}'), load(f'{rerun}_{a}'); rows = []
        for i in f2:
            same = g['same_trace'](f1[i], f2[i])
            if same: continue
            b1 = g['rebuild'](f1[i], g['PARA'][(i, s)], 'first'); b2 = g['rebuild'](f2[i], g['PARA'][(i, s)], 'rerun')
            o1, o2 = g['judge'](g['G0'][i], b1['final']), g['judge'](g['G0'][i], b2['final'])
            rows.append(dict(id=i, first_divergent_turn=diverge(f1[i], f2[i]), final_same=g['fin_key'](b1['final']) == g['fin_key'](b2['final']), outcome_first=o1, outcome_rerun=o2))
        res[f'{name} {a}'] = dict(n=len(f2), differ=len(rows), rows=rows)
if os.path.exists(f'{D}/h1p_diag_warm_twin.jsonl'):
    v1, dg, ct = load('h1v1_seed0_twin'), load('h1p_diag_warm_twin'), load('h1p_control_twin'); rows = []
    for i in dg:
        b = g['rebuild'](dg[i], g['BANK'][i], 'diag'); bv = g['rebuild'](v1[i], g['BANK'][i], 'v1')
        rows.append(dict(id=i, same_as_v1=g['same_trace'](dg[i], v1[i]), same_as_control=g['same_trace'](dg[i], ct[i]),
                         outcome_v1=g['judge'](g['G0'][i], bv['final']), outcome_diag=g['judge'](g['G0'][i], b['final'])))
    res['diagnostic warm twin'] = dict(n=len(dg), same_as_v1=sum(r['same_as_v1'] for r in rows), same_as_control=sum(r['same_as_control'] for r in rows),
                                       outcome_differs_from_v1=[r['id'] for r in rows if r['outcome_v1'] is not None and r['outcome_v1'] != r['outcome_diag']],
                                       differ_from_v1=[r['id'] for r in rows if not r['same_as_v1']])
res['replay_problems'] = g['PROB']
json.dump(res, open(OUT, 'w'), indent=1)
for k, v in res.items():
    if k == 'replay_problems': print(k, len(v)); continue
    print(k, json.dumps({x: y for x, y in v.items() if x != 'rows'})); [print('   ', json.dumps(r)) for r in v.get('rows', [])]
