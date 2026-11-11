#!/usr/bin/env python3
"""azalai_rr_posthoc_masks.py (11 Nov 2026, A100 session) — POST HOC, NOT A RULE: how often m* (rule R's live mask) holds more than
five census points (a canopy-sized SAM 3 mask), per rung, datum (primary, oracle) and twin, at e = 0, and the outcomes then.
python3:  azalai_rr_posthoc_masks.py <out.json> dec_apr1130 dec_apr1400 ..."""
import sys, json
from collections import Counter
O = {}
for s in sys.argv[2:]:
    try: L = [json.loads(l) for l in open(f'/mnt/shared/azalai_rr/records/rr_targets_{s}.jsonl')]
    except FileNotFoundError: continue
    ev = [r for r in L if r['part'] == 2 and 'skip' not in r]; O[s] = {}
    for dn in ('primary', 'oracle'):
        for tw in ('canon', 'bag'):
            x = [r['R'][dn][tw][0] for r in ev]; big = [y for y in x if y.get('n_inside', 0) > 5]
            O[s][f'{dn}_{tw}'] = dict(picks=sum(y['outcome'] != 'no pick' for y in x), mstar_over_5_census_points=len(big), outcomes=dict(Counter(y['outcome'] for y in big)))
            print(s, dn, tw, O[s][f'{dn}_{tw}'])
json.dump(dict(note='POST HOC, NOT A RULE', rungs=O), open(sys.argv[1], 'w'), indent=1)
