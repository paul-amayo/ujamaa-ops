#!/usr/bin/env python3
"""azalai_rr_keynumbers.py (11 Nov 2026, A100 session) — prints the key numbers of each scored rung as sentences (for RESULTS.md and
the final report), straight from rr_summary_<rung>.json, so that no number is retyped. nerf_new python3.10:
  azalai_rr_keynumbers.py dec_apr1400 [dec_apr16_1230 ...]"""
import sys, json
from pathlib import Path
from scipy.stats import binomtest
REC = Path('/mnt/shared/azalai_rr/records')
def ci(k, n): c = binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact'); return c.low, c.high
def fr(k, n): lo, hi = ci(k, n); return f'{k} of {n}, {k / n:.3f} [{lo:.3f}, {hi:.3f}]'
for s in sys.argv[1:]:
    f = REC / f'rr_summary_{s}.json'
    if not f.exists(): print(s, 'not scored'); continue
    S = json.load(open(f)); D = S['part2']; O = D['outcomes']; po = D['primary_outcome']; n = D['evaluable']
    pr = O['all']['primary']['canon']['+0.00']['right plant']; orc = O['all']['oracle']['canon']['+0.00']['right plant']; sq = O['all']['sequence']['canon']['+0.00']['right plant']
    pv = D['paired_e0']['primary_vs_oracle_canon_all']; pvw = D['paired_e0']['primary_vs_oracle_canon_without_datum_failure_lanes']
    w = po['without_datum_failure_lanes']; fail = po['datum_failure_lanes']; tol = {dn: D['tolerance'][dn]['canon']['tolerance_m'] for dn in S['prep']['datums']}
    print(f"== {s}: truth readable {D['truth_readable']} (revisit fit {S['prep']['sitewide_fit_resid_med_m']['revisit']:.4f} m); pairs {D['n_pairs']}, evaluable {n}, not evaluable {D['not_evaluable']}")
    print(f"   primary outcome {fr(pr, n)}, holds {po['all_lanes']['holds_on_logs']}; without datum-failure lanes {fail}: {fr(w['right_plant'], w['n']) if w.get('n') else 'none'}")
    pt = 'n/a' if pvw['sign_test_p'] is None else f"{pvw['sign_test_p']:.2g}"
    print(f"   sequence {fr(sq, n)}; oracle {fr(orc, n)}; primary vs oracle discordant {pv['right_only_primary']} / {pv['right_only_oracle']} (p {pv['sign_test_p']:.2g}); outside failure lanes {pvw['right_only_primary']} / {pvw['right_only_oracle']} (p {pt})")
    print(f"   tolerance {tol}; bag-level twin primary {O['all']['primary']['bag']['+0.00']['right plant']}, oracle {O['all']['oracle']['bag']['+0.00']['right plant']}")
    print(f"   outcomes at e = 0 primary canon: {dict((k, O['all']['primary']['canon']['+0.00'][k]) for k in ('right plant', 'wrong plant', 'unnamed', 'no pick'))}; oracle: {dict((k, O['all']['oracle']['canon']['+0.00'][k]) for k in ('right plant', 'wrong plant', 'unnamed', 'no pick'))}")
    print(f"   per lane right at e = 0 (primary / oracle): " + '; '.join(f"{l}: {v['right_plant_e0']['primary']['canon']}/{v['right_plant_e0']['oracle']['canon']} of {v['n']}{' F' if v['datum_failure'] else ''}" for l, v in D['per_lane'].items()))
    g = D['geometry_only']; print(f"   geometry only partner: primary {g['primary']['all']['partner']}/{g['primary']['all']['n']}, sequence {g['sequence']['all']['partner']}/{g['sequence']['all']['n']}, oracle {g['oracle']['all']['partner']}/{g['oracle']['all']['n']}")
    off = D['pose_offset_px_render_scale'].get('oracle', {}).get('bag_right_plant'); print(f"   pose offset (oracle, bag-level, right plant): {off['median_norm']:.1f} px median, p90 {off['p90_norm']:.1f}, du {off['median_du']:+.1f}, dv {off['median_dv']:+.1f} (n {off['n']})" if off else '   no offset')
