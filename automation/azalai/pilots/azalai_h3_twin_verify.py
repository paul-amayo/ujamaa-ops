#!/usr/bin/env python3
"""azalai_h3_twin_verify.py (11 Nov 2026) — independent recount of the H3 twin pilot's confusion matrices from the per-pair records
alone (h3_twin_pairs.jsonl), re-deciding every arrival from the stored scores rather than trusting the stored verdicts, and compared
with h3_twin_summary.json. Plain python3:  azalai_h3_twin_verify.py <pilots_20261111 dir>"""
import json, sys
from pathlib import Path
D = Path(sys.argv[1]); S = json.load(open(D / 'h3_twin_summary.json')); bad = 0
def decide(rule, x):
    if rule == 'B_bag': return 's_t' in x and x['s_t'] > 0 and all(x['s_t'] > x[k] for k in ('s_prev', 's_next') if x.get(k) is not None)
    return x['iou_t'] > 0 and all(x['iou_t'] > x[k] for k in ('iou_prev', 'iou_next') if x.get(k) is not None)
for season, key in (('apr', 'april_1130'), ('dec', 'december_held_out')):
    R = [json.loads(l) for l in open(D / 'h3_twin_pairs.jsonl')]; R = [r for r in R if r['season'] == season]; ev = [r for r in R if 'skip' not in r]
    print(f'{key}: {len(R)} pairs, {len(ev)} evaluated, {len(R) - len(ev)} not')
    for rule in ('A_canon', 'A_bag', 'B_bag'):
        cc = sum(decide(rule, r['correct'][rule]) for r in ev); wc = sum(decide(rule, r['wrong'][rule]) for r in ev); c = S[key]['all'][rule]
        ok = (len(ev), cc, wc) == (c['n'], c['correct_confirmed'], c['wrong_confirmed']) and all(decide(rule, r[k][rule]) == r[k][rule]['arrived'] for r in ev for k in ('correct', 'wrong')); bad += not ok
        print(f"  {rule:8s} correct confirmed {cc:4d} / rejected {len(ev) - cc:4d} | wrong confirmed {wc:4d} / caught {len(ev) - wc:4d} | summary agrees: {ok}")
    if season == 'apr':
        pc = sum(r['positional']['correct'] for r in R); pw = sum(r['positional']['wrong'] for r in R); print(f'  positional (29 Oct record), all {len(R)}: correct confirmed {pc}, wrong confirmed {pw}; on the evaluated: {sum(r["positional"]["correct"] for r in ev)}, {sum(r["positional"]["wrong"] for r in ev)}')
sys.exit(1 if bad else 0)
