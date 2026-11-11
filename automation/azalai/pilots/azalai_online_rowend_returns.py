#!/usr/bin/env python3
"""azalai_online_rowend_returns.py (A100 agent, 11 Nov 2026): AZ2 post hoc (no verdict). For every read pair and lane, where
the revisit census's row-end bag got its lifted returns: the lane pass's own keyframes, the keyframes between the previous
lane pass and this one (the turn-in), or elsewhere (earlier lanes, later keyframes). Uses datum_diagnostic.json for the
row-end bag ids and part (ii).
  azalai_online_rowend_returns.py <datum_diagnostic.json> <out.json>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azalai_online_geom import load_pass

D = json.load(open(sys.argv[1])); out = []
for e in D:
    rev = e['pair'][len('apr1130_'):] if e['pair'].startswith('apr1130_') else e['pair'][len('dec_'):]
    P = load_pass(rev); lanes = P['lanes']; i = e['lane'] - 1; k0, k1 = lanes[i]['k0'], lanes[i]['k1']; prev_end = lanes[i - 1]['k1'] if i > 0 else -1
    sel = P['ob'] == e['revisit_row_end_bag']; ks, ns = P['ok'][sel], P['on'][sel]; tot = max(int(ns.sum()), 1)
    M = e['M']
    out.append(dict(pair=e['pair'], lane=e['lane'], half=e['half'], part_ii_M=None if M is None else M['part_ii_m'],
                    datum_failure=None if M is None else abs(M['err_first_m']) > e['half'],
                    share_lane_pass=float(ns[(ks >= k0) & (ks <= k1)].sum() / tot), share_turn_in=float(ns[(ks > prev_end) & (ks < k0)].sum() / tot),
                    share_other=float(ns[(ks <= prev_end) | (ks > k1)].sum() / tot), returns=int(ns.sum())))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for o in out:
    if o['part_ii_M'] is not None and abs(o['part_ii_M']) > o['half']:
        print(f"{o['pair']:20s} lane {o['lane']:2d} (ii) {o['part_ii_M']:+.2f} failure {str(o['datum_failure']):5s}: "
              f"lane pass {o['share_lane_pass']:.2f} | turn-in {o['share_turn_in']:.2f} | other {o['share_other']:.2f} of {o['returns']}")
