#!/usr/bin/env python3
"""azalai_online_merge.py (A100 agent, 11 Nov 2026): AZ2 online pilot, the merged online_summary.json and the tables of
RESULTS.md, read from the per-run summaries (online_summary_<tag>.json), the A100 agent's recount and re-simulation records
(recount/recount_<tag>.json, recount/resim_<tag>.json), the verify logs and the post hoc diagnostics. Nothing is recomputed
here except formatting; every number in the tables comes from a file that the recount checked.
  azalai_online_merge.py --records <dir> --recount <dir> --logs <dir> --out <json> --tables <md>"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ('records', 'recount', 'logs', 'out', 'tables'): ap.add_argument('--' + k, required=True)
A = ap.parse_args(); RD, RC, LG = Path(A.records), Path(A.recount), Path(A.logs)
READ = [('dec_apr1130', 'December -> April 15 11:39 (pair of record)'), ('dec_apr1400', 'December -> April 15 14:00'), ('dec_apr16_1230', 'December -> April 16 12:30'),
        ('apr1130_apr1400', 'S1: April 15 11:39 -> April 15 14:00'), ('apr1130_apr16_1230', 'S2: April 15 11:39 -> April 16 12:30')]
UNREAD = [('dec_apr1054', 'December -> April 15 10:54'), ('apr1054_apr1130', 'April 15 10:54 -> 11:39'), ('apr1054_apr1400', 'April 15 10:54 -> 14:00'),
          ('apr1054_apr16_1230', 'April 15 10:54 -> April 16 12:30')]
DIRECT = [(t + '_direct', n + ' (direct-fit truth, sensitivity)') for t, n in UNREAD]


def ld(p):
    return json.load(open(p)) if Path(p).exists() else None


def verify_status(tag):
    f = LG / f'verify_{tag}.log'
    if not f.exists(): return None
    t = f.read_text(); m = re.findall(r'arrivals re-decided without the lane frames, (\w+): (\d+) stops, (\d+) differ', t)
    c = re.search(r'counting re-derived without the lane frames: (\d+) targets, (\d+) differ', t)
    return dict(recount_agrees='recount: ALL AGREE' in t, counting_rederived_differ=int(c.group(2)) if c else None,
                arrivals_rederived_differ={v: [int(n), int(d)] for v, n, d in m})


merged = dict(design='ONLINE_design.md (Azalai 5d76dfe, 19:43:46 UTC); short-gap rungs RR_addendum2_short_gap.md (ad1f663) with design/AZ2_short_gap_note.md (5a4467b, 585175f)',
              runs={})
for tag, name in READ + UNREAD + DIRECT:
    s = ld(RD / f'online_summary_{tag}.json'); rc = ld(RC / f'recount_{tag}.json'); rs = ld(RC / f'resim_{tag}.json')
    if s is None: continue
    keep_vars = {v: {k: s['variants'][v][k] for k in ('n', 'right', 'neighbour', 'other', 'rate', 'ci95', 'no_stop', 'skipped_no_lo', 'abs_miss_med_m', 'abs_miss_p90_m')} | dict(station_error=s['variants'][v]['station_error'], by_lane=s['variants'][v]['by_lane'])
                 for v in s['variants']}
    merged['runs'][tag] = dict(name=name, read_by_rule=bool(s['truth']['readable_by_rule']) and not tag.endswith('_direct'), truth=s['truth'], standoff_d_m=s['standoff_d_m'], spacing_m=s['spacing_m'],
                               half_spacing_m=s['half_spacing_m'], lanes_with_lo=s['lanes_with_lo'], lo_file=s['lo_file'], n_pairs=s['n_pairs'], targets=s['targets'], not_target=s['not_target'],
                               stop_evaluable=s['stop_evaluable'], not_stop_evaluable=s['not_stop_evaluable'], counting=s['counting'], variants=keep_vars, paired=s['paired'],
                               lanes=[dict(lane=l['lane'], sense=l['sense'], **({'skip': l['skip']} if 'skip' in l else {}),
                                           var={v: ({k: l['var'][v][k] for k in ('first_belief_at_m', 'err_first_m', 'datum_failure', 'err_end_m', 'abs_max_m', 'stays_below_half_after_m')} if l.get('var', {}).get(v) and 'err_first_m' in l['var'][v] else None)
                                                for v in (l.get('var') or {})}) for l in s['lanes']],
                               verification=dict(verify_script=verify_status(tag), own_recount_problems=None if rc is None else len(rc['problems']), own_recount_truth=None if rc is None else rc['truth'],
                                                 own_resim_issues=None if rs is None else len(rs['issues'])))
merged['post_hoc'] = dict(datum_diagnostic=ld(RC / 'datum_diagnostic.json'), row_end_bag_returns=ld(RC / 'rowend_returns.json'), laser_odometry_vs_graph_travel=ld(RC / 'odo_vs_graph.json'))
json.dump(merged, open(A.out, 'w'), indent=1)

# ------------------------------------------------------------------ tables
R = merged['runs']; L = []
f3 = lambda x: f'{x:.3f}'; ci = lambda v: f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
def rr(v): return f"{v['right']}/{v['n']} = {v['rate']:.3f}" if v['n'] else 'n/a'
L.append('### Table 1. Right-plant rates over the stop-evaluable targets (read pairs)\n')
L.append('| pair | fit residuals (m) | pairs / targets / stop-evaluable | counting | **M (primary)** [95 %] | a dead reckoning | b graph poses | cp | cm | S (secondary) | oracle |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
for tag, name in READ:
    r = R[tag]; v = r['variants']; c = r['counting']['stop_evaluable']; tr = r['truth']['sitewide_resid_med_m']
    L.append(f"| {name} | {tr['registry']:.3f} / {tr['revisit']:.3f} | {r['n_pairs']} / {r['targets']} / {r['stop_evaluable']} | {rr(c)} | **{rr(v['M'])}** {ci(v['M'])} | {rr(v['a'])} | {rr(v['b'])} | {rr(v['cp'])} | {rr(v['cm'])} | {rr(v['S'])} | {rr(v['O'])} |")
L.append('\n### Table 2. Paired exact sign tests on the same stop-evaluable targets (M right only / other right only, two-sided p)\n')
L.append('| pair | M vs counting | M vs a | M vs b | S vs M (secondary) |'); L.append('|---|---|---|---|---|')
for tag, name in READ:
    p = R[tag]['paired']; fm = lambda k: f"{p[k]['first_only']} / {p[k]['second_only']}, p = {p[k]['p']:.2g}"
    L.append(f"| {name} | {fm('M_vs_counting')} | {fm('M_vs_a')} | {fm('M_vs_b')} | {fm('S_vs_M')} |")
L.append('\n### Table 3. Station error over all lane keyframes with a belief: |e| median / 90th percentile / maximum (m), share above S/2, datum failures (lanes whose first |e| > S/2)\n')
L.append('| pair | S/2 (m) | M | a | b |'); L.append('|---|---|---|---|---|')
for tag, name in READ:
    r = R[tag]; cell = lambda v: (lambda se: f"{se['abs_med_m']:.2f} / {se['abs_p90_m']:.2f} / {se['abs_max_m']:.2f}; {se['share_over_half']:.2f}; lanes {', '.join(map(str, se['datum_failures'])) or 'none'}")(r['variants'][v]['station_error'])
    L.append(f"| {name} | {r['half_spacing_m']:.3f} | {cell('M')} | {cell('a')} | {cell('b')} |")
L.append('\n### Table 4. Station error of M by distance into the lane, 5 m bands: |e| median / share above S/2 (keyframes)\n')
bands = list(R['dec_apr1130']['variants']['M']['station_error']['bands_5m'])
L.append('| pair | ' + ' | '.join(bands) + ' |'); L.append('|---|' + '---|' * len(bands))
for tag, name in READ:
    b = R[tag]['variants']['M']['station_error']['bands_5m']
    L.append(f'| {name} | ' + ' | '.join('' if b[k] is None else f"{b[k]['abs_med_m']:.2f} / {b[k]['share_over_half']:.2f} ({b[k]['n']})" for k in bands) + ' |')
L.append('\n### Table 5. Error at the first belief per lane, M (m); * = datum failure (|e| > S/2)\n')
L.append('| pair | ' + ' | '.join(f'lane {i}' for i in range(1, 11)) + ' |'); L.append('|---|' + '---|' * 10)
for tag, name in READ:
    r = R[tag]; h = r['half_spacing_m']; cells = []
    for l in r['lanes']:
        v = l['var'].get('M') if l.get('var') else None
        cells.append('' if v is None else f"{v['err_first_m']:+.2f}{'*' if v['datum_failure'] else ''}")
    L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
L.append('\n### Table 6. Wrong row end (c): distance into the lane after which |e| stays below S/2 to the lane\'s end (m; "never")\n')
L.append('| pair | sign | ' + ' | '.join(f'lane {i}' for i in range(1, 11)) + ' | lanes recovering |'); L.append('|---|---|' + '---|' * 11)
for tag, name in READ:
    for v in ('cp', 'cm'):
        cells = []; n = 0
        for l in R[tag]['lanes']:
            x = l['var'].get(v) if l.get('var') else None
            if x is None: cells.append(''); continue
            st = x['stays_below_half_after_m']; cells.append('never' if st == 'never' else f'{st:.1f}'); n += st != 'never'
        L.append(f"| {name} | {'+S' if v == 'cp' else '-S'} | " + ' | '.join(cells) + f' | {n} of 10 |')
L.append('\n### Table 7. Counting baseline by the registry rank n (all targets): right / n, median |count offset|\n')
L.append('| pair | all targets | 1-10 | 11-20 | 21-30 | 31-40 | 41 and above |'); L.append('|---|---|---|---|---|---|---|')
for tag, name in READ:
    c = R[tag]['counting']; bn = c['by_n']
    cell = lambda k: '' if not bn[k]['n'] else f"{bn[k]['right']}/{bn[k]['n']}, {bn[k]['abs_offset_med']:g}"
    L.append(f"| {name} | {rr(c['all_targets'])} | " + ' | '.join(cell(k) for k in ('1-10', '11-20', '21-30', '31-40', '41-up')) + ' |')
L.append('\n### Table 8. Truth too loose to read (April 15 10:54 site-wide fit 0.1517 m > 0.15 m): counts only, no verdict\n')
L.append('| pair | pairs / targets / stop-evaluable | counting | M | a | b | S | oracle | note |'); L.append('|---|---|---|---|---|---|---|---|---|')
for tag, name in UNREAD + DIRECT:
    if tag not in R: continue
    r = R[tag]; v = r['variants']; note = f"direct fit residual {r['truth']['direct_resid_med_m']:.3f} m" if tag.endswith('_direct') else (f"laser odometry lane(s) {r['lanes_with_lo']}" if len(r['lanes_with_lo']) < 10 else '')
    L.append(f"| {name} | {r['n_pairs']} / {r['targets']} / {r['stop_evaluable']} | {rr(r['counting']['stop_evaluable'])} | {rr(v['M'])} | {rr(v['a'])} | {rr(v['b'])} | {rr(v['S'])} | {rr(v['O'])} | {note} |")
L.append('\n### Table 9. Verification per run\n')
L.append('| run | verify script: recount | counting re-derived, differ | arrivals re-decided, differ (M / a / b) | own recount: disagreements | own re-simulation: disagreements |'); L.append('|---|---|---|---|---|---|')
for tag, name in READ + UNREAD + DIRECT:
    if tag not in R: continue
    vf = R[tag]['verification']; vs = vf['verify_script'] or {}; ad = vs.get('arrivals_rederived_differ', {})
    arr = ' / '.join(f"{ad[k][1]} of {ad[k][0]}" for k in ('M', 'a', 'b') if k in ad)
    L.append(f"| {tag} | {'agrees' if vs.get('recount_agrees') else 'n/a'} | {vs.get('counting_rederived_differ')} | {arr} | {vf['own_recount_problems']} | {'not run (direct truth)' if vf['own_resim_issues'] is None else vf['own_resim_issues']} |")
Path(A.tables).write_text('\n'.join(L) + '\n'); print('\n'.join(L))
