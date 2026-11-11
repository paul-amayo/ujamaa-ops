#!/usr/bin/env python3
"""azalai_rr_tables.py (11 Nov 2026, A100 session) — the Markdown tables of RESULTS.md, generated from the records alone (no number
is typed by hand): per rendered rung the outcome counts at every e (datum x twin), the subsets, the geometry-only step, the per-lane
datum table, the paired tests; Part 1 and Part 3 for the pair of record; the ladder table; the geometry-only rungs. nerf_new:
  azalai_rr_tables.py <out.md>"""
import sys, json, numpy as np
from pathlib import Path
from scipy.stats import binomtest
REC = Path('/mnt/shared/azalai_rr/records'); GEO = Path('/mnt/shared/azalai_rr/geom'); B = Path('/mnt/shared/azalai_rr'); out = []
def ci(k, n): c = binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact'); return c.low, c.high
def fr(k, n): lo, hi = ci(k, n); return f'{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]'
def P(s=''): out.append(s)
def rung_tables(s, title, full=True):
    S = json.load(open(REC / f'rr_summary_{s}.json')); D = S['part2']; O = D['outcomes']; DAT = S['prep']['datums']; ES = sorted(float(e) for e in O['all']['primary']['canon'])
    P(f'### {title}'); P()
    P(f"{D['evaluable']} of {D['n_pairs']} pairs evaluable; not evaluable: " + '; '.join(f'{k} {v}' for k, v in D['not_evaluable'].items()) + f". Site-wide fit residual medians: twin {S['prep']['sitewide_fit_resid_med_m']['twin']:.3f} m, revisit {S['prep']['sitewide_fit_resid_med_m']['revisit']:.4f} m (truth readable: {D['truth_readable']}). Standoff d = {S['prep']['standoff_d_m']:.3f} m; median inner-row bag spacing {S['prep']['spacing_m']:.3f} m (datum failure over {S['prep']['half_spacing_m']:.3f} m); datum-failure lanes {D['primary_outcome']['datum_failure_lanes']}."); P()
    P('Outcomes of rule R at every e (right plant / wrong plant / unnamed / no pick), all evaluable pairs:'); P()
    P('| datum | twin | ' + ' | '.join(f'e = {e:+.2f}' for e in ES) + ' | tolerance (m) |'); P('|---|---|' + '---|' * (len(ES) + 1))
    for dn in DAT:
        for tw in ('canon', 'bag'):
            c = O['all'][dn][tw]; P(f"| {dn} | {'letter' if tw == 'canon' else 'bag-level'} | " + ' | '.join(f"{c[f'{e:+.2f}']['right plant']} / {c[f'{e:+.2f}']['wrong plant']} / {c[f'{e:+.2f}']['unnamed']} / {c[f'{e:+.2f}']['no pick']}" for e in ES) + f" | {D['tolerance'][dn][tw]['tolerance_m']:.2f} |")
    P(); P('Right-plant rate at e = 0 with exact 95 % interval, by subset (twin of the letter; bag-level twin in brackets as a count):'); P()
    P('| subset | n | ' + ' | '.join(DAT) + ' |'); P('|---|---|' + '---|' * len(DAT))
    for name in ('all', 'without_datum_failure_lanes', 'datum_failure_lanes', 'naming_confirmed', 'naming_not_confirmed', 'growth_stable', 'growth_changed', 'growth_no_call'):
        c0 = O[name]['primary']['canon']['+0.00']; n = c0['n']
        if not n: P(f'| {name.replace("_", " ")} | 0 |' + ' |' * len(DAT)); continue
        P(f"| {name.replace('_', ' ')} | {n} | " + ' | '.join(f"{fr(O[name][dn]['canon']['+0.00']['right plant'], n)} ({O[name][dn]['bag']['+0.00']['right plant']})" for dn in DAT) + ' |')
    P(); P('The same outcome counts at every e for each subset (twin of the letter; right / wrong / unnamed / no pick):'); P()
    P('| subset | datum | ' + ' | '.join(f'e = {e:+.2f}' for e in ES) + ' |'); P('|---|---|' + '---|' * len(ES))
    for name in ('without_datum_failure_lanes', 'datum_failure_lanes', 'naming_confirmed', 'naming_not_confirmed', 'growth_stable', 'growth_changed', 'growth_no_call'):
        for dn in DAT:
            c = O[name][dn]['canon']
            if c['+0.00']['n']: P(f"| {name.replace('_', ' ')} (n {c['+0.00']['n']}) | {dn} | " + ' | '.join(f"{c[f'{e:+.2f}']['right plant']} / {c[f'{e:+.2f}']['wrong plant']} / {c[f'{e:+.2f}']['unnamed']} / {c[f'{e:+.2f}']['no pick']}" for e in ES) + ' |')
    P(); P('Paired at e = 0 (right plant under one datum and not the other; exact two-sided sign test):'); P()
    P('| comparison | twin | subset | n | only first | only second | both | p |'); P('|---|---|---|---|---|---|---|---|')
    for k, v in D['paired_e0'].items():
        x, rest = k.split('_vs_'); y, tw, name = rest.split('_', 2)
        if not (x == 'primary' or (x == 'sequence' and y == 'oracle') or (x == 'oracle' and y == 'sitewide')): continue
        P(f"| {x} vs {y} | {'letter' if tw == 'canon' else 'bag-level'} | {name.replace('_', ' ')} | {v['n']} | {v[f'right_only_{x}']} | {v[f'right_only_{y}']} | {v['both']} | {v['sign_test_p']:.2g} |")
    P(); P('Geometry only (step 5): the target carried into the revisit by the datum\'s transfer; nearest revisit bag is the partner:'); P()
    P('| datum | all | without datum-failure lanes | datum-failure lanes | distance to partner, median / p90 (m), all |'); P('|---|---|---|---|---|')
    for dn in DAT:
        g = D['geometry_only'][dn]; P(f"| {dn} | {g['all']['partner']}/{g['all']['n']} | {g['without_datum_failure_lanes']['partner']}/{g['without_datum_failure_lanes']['n']} | {g['datum_failure_lanes']['partner']}/{g['datum_failure_lanes']['n']} | {g['all']['partner_dist_median_m']:.2f} / {g['all']['partner_dist_p90_m']:.2f} |")
    P(); P('Per lane: datum, transfer against the site-wide fit at the lane\'s middle (primary datum), and right-plant counts at e = 0 (twin of the letter):'); P()
    Pp = json.load(open(B / s / 'rr_prep_summary.json')); LN = {l['lane']: l for l in Pp['lanes']}
    P('| lane | sense | evaluable | oracle row end in the primary frame (m) | sequence (m) | datum failure | along / lateral / vertical (m), rotation (deg) | ' + ' | '.join(f'right, {dn}' for dn in DAT) + ' | geometry partner, primary |'); P('|---|---|---|---|---|---|---|' + '---|' * len(DAT) + '---|')
    for l in range(1, 11):
        L = LN[l]; v = L['vs_sitewide']['primary']; pl = D['per_lane'].get(str(l))
        P(f"| {l} | {L['sense']:+d} | {pl['n'] if pl else 0} | {L['row_end_minus_oracle_m']['primary']:+.2f} | {L['row_end_minus_oracle_m']['sequence']:+.2f} | {'yes' if L['datum_failure'] else 'no'} | {v['middle']['along_m']:+.2f} / {v['middle']['lateral_m']:+.2f} / {v['middle']['vertical_m']:+.2f}, {v['rotation_deg']:.2f} | "
          + ' | '.join(str(pl['right_plant_e0'][dn]['canon']) if pl else '-' for dn in DAT) + f" | {pl['geometry_partner']['primary'] if pl else '-'} |")
    P(); off = D['pose_offset_px_render_scale']
    P('Pose transfer check (b): centroid of m* minus the largest component of M_t at e = 0, pixels at 640 x 360, right-plant outcomes (median norm, p90, median du, dv):'); P()
    P('| datum | twin of the letter | bag-level twin |'); P('|---|---|---|')
    for dn in DAT:
        a_, b_ = off[dn].get('canon_right_plant'), off[dn].get('bag_right_plant'); f = lambda x: f"n {x['n']}: {x['median_norm']:.1f} px, p90 {x['p90_norm']:.1f}, ({x['median_du']:+.1f}, {x['median_dv']:+.1f})" if x else '-'
        P(f'| {dn} | {f(a_)} | {f(b_)} |')
    P(); return S
S0 = rung_tables('dec_apr1130', 'Pair of record: December twin, revisit April 15 11:39 (apr1130), the 552 pilot pairs')
# Part 1
D1 = S0['part1']; P('### Part 1: relevancy from free poses in the twin (December only)'); P()
P(f"{D1['n_targets']} targets (anchors in a cell's vocabulary within 1.5 m of its lane line), {D1['census_inner']} of them named census bags of the lane's inner rows."); P()
P('| twin | targets | pose | n | lit | mask area median (px) | lands on the target | landing error median / p90 (m) | range median (m) | abs bearing median (deg) | to nearest training camera median / p90 (m) | heading to it median (deg) | station miss median (m) |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
tagn = dict(C='free, centre line', L='free, 0.3 m left', R='free, 0.3 m right', H='held-out keyframe', T='training keyframe')
for tw in ('canon', 'bag'):
    for name in ('all', 'census_inner'):
        for tag in 'CLRHT':
            x = D1['poses'][f'{tw}_{name}_{tag}']; sm = x['station_miss_m_median']
            P(f"| {'letter' if tw == 'canon' else 'bag-level'} | {name.replace('_', ' ')} | {tagn[tag]} | {x['n']} | {x['lit']} | {x['area_px_median']:.0f} | {x['lands']} | {x['err_m_median']:.2f} / {x['err_m_p90']:.2f} | {x['range_m_median']:.2f} | {x['abs_bearing_deg_median']:.1f} | {x['train_dist_m_median']:.2f} / {x['train_dist_m_p90']:.2f} | {x['train_heading_deg_median']:.1f} | {'-' if sm is None else f'{sm:.2f}'} |")
P(); P('Paired (lands on the target; exact two-sided sign test):'); P(); P('| twin | targets | comparison | n | lands first | lands second | only first | only second | p |'); P('|---|---|---|---|---|---|---|---|---|')
for k, v in D1['paired'].items():
    tw, rest = k.split('_', 1); name, cmp_ = ('census_inner', rest[len('census_inner_'):]) if rest.startswith('census_inner_') else ('all', rest[len('all_'):]); x, y = cmp_.split('_vs_')
    P(f"| {'letter' if tw == 'canon' else 'bag-level'} | {name.replace('_', ' ')} | {tagn[x]} vs {tagn[y]} | {v['n']} | {v[f'lands_{x}']} | {v[f'lands_{y}']} | {v[f'only_{x}']} | {v[f'only_{y}']} | {v['sign_test_p']:.2g} |")
P(); P('Gaussians whose own feature reads the target\'s address, distance of their median position from the anchor in the ground plane:'); P()
for tw, g in D1['gauss'].items(): P(f"- {'twin of the letter' if tw == 'canon' else 'bag-level twin'}: {g['with_gaussians']} of {g['n']} targets have such Gaussians (median {g['n_gaussians_median']:.0f}); distance median {g['dist_ground_m_median']:.2f} m, p90 {g['dist_ground_m_p90']:.2f} m; within half a bag spacing: {g['within_half_spacing']}.")
P()
# Part 3
O3 = json.load(open(REC / 'rr_odo.json')); P('### Part 3: along-row odometry error on the logs (laser odometry `lo_poses.npz`, plain, against the INS)'); P()
P('| survey | lane | sense | seconds | L end (m) | e_odo end (m) | e_odo max (m) | at L (m) | largest lidar gap (ms) | INS fix records (antenna 1 min, RTK fixed) | usable |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
for name in ('dec', 'apr1130'):
    for r in O3['surveys'][name]['passes']:
        p = r['plain']; f = r['fix']; P(f"| {name} | {r['lane']} | {r['sense']:+d} | {r['seconds']:.0f} | {p['L_end_m']:.1f} | {p['e_end_m']:.3f} | {p['e_max_m']:.3f} | {p['L_at_max_m']:.1f} | {p['lo_gap_max_ms']:.0f} | {f['records']} ({f['antenna1_min']}, {f['antenna1_rtk_fixed']}) | {r['usable']} |")
s3 = O3['summary']['plain']; P(); P(f"Over the {s3['passes']} passes: e_odo at the lane's end median {s3['e_end_m']['median']:.3f} m, 90th percentile {s3['e_end_m']['p90']:.3f} m, maximum {s3['e_end_m']['max']:.3f} m; maximum over the lane median {s3['e_max_m']['median']:.3f} m, p90 {s3['e_max_m']['p90']:.3f} m, maximum {s3['e_max_m']['max']:.3f} m. Left out: {s3['left_out'] or 'none'}.")
for v in ('viopatch', 'viopatch_icp'):
    w = O3['summary'].get(f'{v}_where_it_exists')
    if w: P(f"With `lo_poses_{v}.npz` where it exists (April 11:39; plain elsewhere): e_odo end median {w['e_end_m']['median']:.3f} m, p90 {w['e_end_m']['p90']:.3f} m, maximum {w['e_end_m']['max']:.3f} m.")
for name in ('apr1400', 'apr16_1230'):
    w = O3['summary']['per_survey'].get(name)
    if w: P(f"Extra pass {name} (not one of the design's 20): e_odo end median {w['e_end_m']['median']:.3f} m, p90 {w['e_end_m']['p90']:.3f} m, maximum {w['e_end_m']['max']:.3f} m over {w['passes']} passes.")
P()
# the other rendered rungs
for s, t in (('dec_apr1400', 'D2 rung: December twin, revisit April 15 14:00 (apr1400), every mutual nearest neighbour pair'), ('dec_apr16_1230', 'D2 rung: December twin, revisit April 16 12:30 (apr16_1230), every mutual nearest neighbour pair'),
             ('dec_apr1130_allmnn', 'Sensitivity (not a rule): pair of record over every mutual nearest neighbour pair'), ('dec_apr1054', 'D2 rung apr1054: truth too loose to read'), ('dec_feb1003', 'D1 rung feb1003: truth too loose to read'), ('dec_feb1224', 'D1 rung feb1224: truth too loose to read')):
    if (REC / f'rr_summary_{s}.json').exists(): rung_tables(s, t)
# the ladder
P('### The ladder (right-plant rate of rule R at e = 0, twin of the letter, exact 95 % interval)'); P()
P('| rung | twin | revisit | gap | read | evaluable | primary | sequence | oracle | tolerance, primary (m) | verdict by the 0.80 rule |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
for s, rung, rev, gap in (('dec_apr1130', 'D2 (pair of record)', 'Apr 15 11:39', '4 months'), ('dec_apr1400', 'D2', 'Apr 15 14:00', '4 months'), ('dec_apr16_1230', 'D2', 'Apr 16 12:30', '4 months'), ('dec_apr1054', 'D2', 'Apr 15 10:54', '4 months'), ('dec_feb1003', 'D1', 'Feb 10:03', '2 months'), ('dec_feb1224', 'D1', 'Feb 12:24', '2 months')):
    f = REC / f'rr_summary_{s}.json'
    if not f.exists(): P(f'| {rung} | December | {rev} | {gap} | not run | - | - | - | - | - | - |'); continue
    S = json.load(open(f)); D = S['part2']; O = D['outcomes']['all']; n = D['evaluable']; rd = D['truth_readable']; po = D['primary_outcome']['all_lanes']
    P(f"| {rung} | December | {rev} | {gap} | {'yes' if rd else 'no (truth too loose)'} | {n} | " + ' | '.join(fr(O[dn]['canon']['+0.00']['right plant'], n) for dn in ('primary', 'sequence', 'oracle')) + f" | {D['tolerance']['primary']['canon']['tolerance_m']:.2f} | {('holds' if po['holds_on_logs'] else 'does not hold') if rd else 'not read'} |")
for rung, rev, gap in (('A1', 'Apr 15 11:39', '45 minutes'), ('A2', 'Apr 15 14:00', '3 hours'), ('A3', 'Apr 16 12:30', '1 day')): P(f'| {rung} | April 15 10:54 | {rev} | {gap} | not run (addendum provision) | - | - | - | - | - | - |')
P()
# the geometry-only rungs and comparators
P('### Geometry-only rungs (addendum 2) and the December comparators on the same construction (step 5, every inner-row pair, no twin)'); P()
P('| registry -> revisit | read | pairs | evaluable | datum-failure lanes | sequence over half a spacing | step 5 partner, primary | without datum-failure lanes | sequence | oracle | standoff d (m) | spacing (m) |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for s in ('apr1130_apr1400', 'apr1130_apr16_1230', 'dec_apr1130_allmnn', 'dec_apr1400_allmnn', 'dec_apr16_1230_allmnn'):
    G = json.load(open(GEO / s / f'rr_geom_{s}.json'))['summary']; g = G['geometry_only']
    P(f"| {G['registry']} -> {G['revisit']} | {'yes' if G['truth_readable'] else 'no'} | {G['n_pairs']} | {G['evaluable']} | {G['datum_failures']} | {G['sequence_off_lanes']} | {fr(g['primary']['all']['partner'], g['primary']['all']['n'])} | {g['primary']['without_datum_failure_lanes']['partner']}/{g['primary']['without_datum_failure_lanes']['n']} | {g['sequence']['all']['partner']}/{g['sequence']['all']['n']} | {g['oracle']['all']['partner']}/{g['oracle']['all']['n']} | {G['standoff_d_m']:.3f} | {G['spacing_m']:.3f} |")
P(); P('Per-lane row-end offsets (oracle row end in the primary frame, m; * = datum failure):'); P()
P('| registry -> revisit | ' + ' | '.join(f'lane {i}' for i in range(1, 11)) + ' |'); P('|---|' + '---|' * 10)
for s in ('dec_apr1130_allmnn', 'dec_apr1400_allmnn', 'dec_apr16_1230_allmnn', 'apr1130_apr1400', 'apr1130_apr16_1230'):
    G = json.load(open(GEO / s / f'rr_geom_{s}.json'))['summary']; L = {l['lane']: l for l in G['lanes']}
    P(f"| {G['registry']} -> {G['revisit']} | " + ' | '.join(f"{L[i]['row_end_minus_oracle_m']['primary']:+.2f}{'*' if L[i]['datum_failure'] else ''}" for i in range(1, 11)) + ' |')
P(); open(sys.argv[1], 'w').write('\n'.join(out) + '\n'); print('[tables]', sys.argv[1], len(out), 'lines')
