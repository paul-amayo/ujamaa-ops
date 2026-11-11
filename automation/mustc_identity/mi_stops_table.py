#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3), A100 session 11-11: per-stop markdown tables for RESULTS.md, read straight from each stop's records
(recount.json from mi_recount.py, heights_summary.json from mi_heights.py, verify_heights.json from mi_verify_heights.py, build_summary.json,
registration.json). No pooling across stops (the design has no pooling rule).
Any python3:  mi_stops_table.py <stop dir> [<stop dir> ...]"""
import json, os, sys
REG = json.load(open('/mnt/shared2/must-c/plot198/census/registration.json'))
f3 = lambda x: '-' if x is None else f'{x:.3f}'
L1 = ['| stop | registration: soil residual / plant-pattern check | masks: labelled / ambiguous / off-row / no point | nodes with census masks (views median) | held-out labels nikon_5 / nikon_15 |', '|---|---|---|---|---|']
L2 = ['| stop | view | plant, size rule: median IoU (n) | plant, all (n) | row, size rule (n) | row, all (n) | nesting (frame / per plant) | recount max abs dIoU |', '|---|---|---|---|---|---|---|---|']
L3 = ['| stop | S | address: found, bias, med abs, p95 | c1: found, bias, med abs, p95 | c2: found, bias, med abs, p95 | all three: n; bias address / c1 / c2 | verifier differences |', '|---|---|---|---|---|---|---|']
for D in sys.argv[1:]:
    st = os.path.basename(os.path.normpath(D)); stop = '230613_' + st
    b = json.load(open(f'{D}/cell/build_summary.json')); r = REG[stop]; ok = r['registration_ok']; g = r['gate']
    mb = b['masks_by_outcome']; v = list(b['census_views_per_labelled_node'].values()); v.sort()
    ho = b['nodes_with_mask_in_held_out_views']; n5 = sorted(int(k) for k, x in ho.items() if 'nikon_5' in x); n15 = sorted(int(k) for k, x in ho.items() if 'nikon_15' in x)
    L1.append(f"| {st} | {ok['soil_residual_median_m'] * 100:.2f} cm / {'passed' if g['pass'] else 'not passed'} ({g['n_matched']} of {g['n_splat']}, {g['residual_median_m'] * 100:.1f} cm) | {mb['labelled']} / {mb['ambiguous']} / {mb['off_row_only']} / {mb['no_census_point']} | {len(v)} of {b['nodes_on_row']} ({v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2:g}) | {len(n5)} / {len(n15)} |")
    R = json.load(open(f'{D}/recount.json')); rd = R['rederived_from_pixels']   # medians below are exact, from the per-head records (recount.json rounds to 4 decimals)
    recs = [json.loads(l) for l in open(f'{D}/identity/protocol_tangent_vf.jsonl')]
    grp = lambda fr: 'nikon_5' if fr.startswith('nikon_5.') else ('nikon_15' if fr.startswith('nikon_15.') else 'census_views')
    import statistics as _st
    for view in ('nikon_5', 'nikon_15', 'census_views'):
        def c(k, rule):
            v = [x['argmax']['iou'] for x in recs if x['kind'] == k and grp(x['frame']) == view and x['gt_frac'] >= (0.002 if rule == 'ge_0.2pct' else 0.0)]
            return {'iou_median': _st.median(v) if v else None, 'n': len(v)}
        cell = lambda k, rule: f"{f3(c(k, rule).get('iou_median'))} ({c(k, rule).get('n', 0)})"
        nf = [x['nesting'] for x in recs if x['kind'] == 'FRAME' and grp(x['frame']) == view and x['nesting'] == x['nesting']]
        npl = [x['nest_pred'] for x in recs if x['kind'] == 'TREE' and grp(x['frame']) == view and 'nest_pred' in x]
        fr = {'nesting_median': _st.median(nf) if nf else None}; npm = _st.median(npl) if npl else None
        L2.append(f"| {st} | {view} | {cell('TREE', 'ge_0.2pct')} | {cell('TREE', 'all')} | {cell('ROW', 'ge_0.2pct')} | {cell('ROW', 'all')} | {f3(fr.get('nesting_median'))} / {f3(npm)} | {rd['max_abs_iou_difference']:.4f} ({rd['heads_rederived']} of {rd['heads_in_records']}) |")
    H = json.load(open(f'{D}/heights_summary.json')); V = json.load(open(f'{D}/verify_heights.json'))
    def hs(x): return f"{x['found']}, {x.get('bias_median_m', 0) * 100:+.1f} cm, {x.get('absdiff_median_m', 0) * 100:.1f} cm, {x.get('absdiff_p95_m', 0) * 100:.1f} cm" if x.get('n') else f"{x['found']}, -, -, -"
    p3 = H['paired_all_three']
    tri = f"{p3['n']}; " + (' / '.join(f"{p3[k]['bias_median_m'] * 100:+.1f}" for k in ('address', 'c1', 'c2')) + ' cm' if p3['n'] else '-')
    L3.append(f"| {st} | {H['S']['n']} ({H['S']['with_census_supervision']} supervised) | {hs(H['address'])} | {hs(H['c1_splat_census_b'])} | {hs(H['c2_laser_footprint_record'])} | {tri} | {len(V['differences_vs_heights_csv'])} |")
print('\n'.join(L1)); print(); print('\n'.join(L2)); print(); print('\n'.join(L3))
