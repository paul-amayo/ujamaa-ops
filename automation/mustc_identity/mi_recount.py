#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3): independent recount. Shares no code with argmax_eval.py or mi_heights.py.
 A. protocol table from the per-head records (protocol_tangent_vf.jsonl): medians per split / view group / level, with and without
    the letter's size rule (reference >= 0.2 % of the frame), nesting per frame and per plant.
 B. the same IoUs re-derived from the dumped per-pixel decisions (pred_tangent_vf/<frame>.npz keys tree, row) against the supervision
    id maps, pixel by pixel, and the largest difference to the records.
 C. the height table from heights.csv.
Box python with numpy + PIL:  mi_recount.py --dir <stop dir> --identity-root <identity dir with marker_hierarchy.json> [--out recount.json]"""
import argparse, csv, json, statistics as stt
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--identity-root', required=True); ap.add_argument('--tag', default='tangent_vf'); ap.add_argument('--out', default=''); a = ap.parse_args(); D = a.dir
recs = [json.loads(l) for l in open(f'{D}/identity/protocol_{a.tag}.jsonl')]
heads = [r for r in recs if r['kind'] in ('TREE', 'ROW')]; frames = {r['frame']: r for r in recs if r['kind'] == 'FRAME'}
def med(v): return round(stt.median(v), 4) if v else None
def group(fr): return 'nikon_5' if fr.startswith('nikon_5.') else ('nikon_15' if fr.startswith('nikon_15.') else 'census_views')
A = {}
for g in ('nikon_5', 'nikon_15', 'census_views'):
    for kind in ('TREE', 'ROW'):
        for rule, lo in (('ge_0.2pct', 0.002), ('all', 0.0)):
            hh = [r for r in heads if group(r['frame']) == g and r['kind'] == kind and r['gt_frac'] >= lo]
            A[f'{g}|{kind}|{rule}'] = {'n': len(hh), 'iou_median': med([r['argmax']['iou'] for r in hh]), 'prec_median': med([r['argmax']['prec'] for r in hh]), 'rec_median': med([r['argmax']['rec'] for r in hh]),
                                       'iou_min': round(min([r['argmax']['iou'] for r in hh]), 4) if hh else None, 'iou_max': round(max([r['argmax']['iou'] for r in hh]), 4) if hh else None,
                                       'oracle_iou_median': med([r['oracle']['iou'] for r in hh]), 'gt_void_frac_median': med([r['gt_void_frac'] for r in hh]),
                                       'nest_pred_median': med([r['nest_pred'] for r in hh if 'nest_pred' in r]) if kind == 'TREE' else None,
                                       'sibling_confusion_median': med([r['sibling_confusion'] for r in hh if 'sibling_confusion' in r]) if kind == 'TREE' else None,
                                       'argmax_union_iou_median': med([r['argmax_union']['iou'] for r in hh if 'argmax_union' in r]) if kind == 'ROW' else None}
    fr = [f for n, f in frames.items() if group(n) == g]
    A[f'{g}|FRAME'] = {'n': len(fr), 'nesting_median': med([f['nesting'] for f in fr if f['nesting'] == f['nesting']]), 'void_frac_median': med([f['void_frac'] for f in fr])}
# B: re-derive from the dumped decisions
H = json.load(open(f'{a.identity_root}/marker_hierarchy.json')); row_of = {o['id']: o['row_id'] for o in H['objects']}; maxd = 0.0; nB = 0; worst = None; B = []
for name in frames:
    pr = np.load(f'{D}/pred_{a.tag}/{name}.npz'); tree, row = pr['tree'], pr['row']; hh, ww = tree.shape
    gt = np.array(Image.fromarray(np.array(Image.open(f'{D}/cell/supervision/trees_only/{name}'), np.uint16)).resize((ww, hh), Image.NEAREST)).astype(np.int64)
    ids = [int(u) for u in np.unique(gt) if u != 65535]
    cases = [('TREE', i, gt == i, tree == i) for i in ids] + [('ROW', r, np.isin(gt, [i for i in ids if row_of.get(i) == r]), row == r) for r in sorted({row_of[i] for i in ids})]
    for kind, i, g, p in cases:
        inter = int((g & p).sum()); un = int((g | p).sum()); iou = inter / max(un, 1); nB += 1
        ref = next((r for r in heads if r['frame'] == name and r['kind'] == kind and r['id'] == i), None)
        dd = abs(iou - ref['argmax']['iou']) if ref else float('nan'); B.append({'frame': name, 'kind': kind, 'id': i, 'iou_rederived': round(iou, 4), 'iou_record': ref['argmax']['iou'] if ref else None, 'gt_px': int(g.sum()), 'pred_px': int(p.sum())})
        if ref is None or dd > maxd: maxd = dd if ref else float('inf'); worst = B[-1]
A_B = {'heads_rederived': nB, 'heads_in_records': len(heads), 'max_abs_iou_difference': maxd, 'worst': worst}
# C: heights
out = {'protocol_from_records': A, 'rederived_from_pixels': A_B}
try:
    rows = list(csv.DictReader(open(f'{D}/heights.csv'))); Cc = {'n_S': len(rows)}
    for key in ('address_h95_m', 'c1_h95_m', 'c2_h95_m'):
        dh = sorted(float(r[key]) - float(r['laser_h95_m']) for r in rows if r[key] != ''); ad = sorted(abs(x) for x in dh)
        Cc[key] = {'found': len(dh), 'bias_median_m': med(dh), 'absdiff_median_m': med(ad), 'absdiff_p95_m': round(float(np.percentile(ad, 95)), 4) if ad else None, 'absdiff_max_m': round(max(ad), 4) if ad else None}
    both = [r for r in rows if r['address_h95_m'] != '' and r['c1_h95_m'] != '']
    Cc['paired_address_c1'] = {'n': len(both), 'address_bias_median_m': med([float(r['address_h95_m']) - float(r['laser_h95_m']) for r in both]), 'c1_bias_median_m': med([float(r['c1_h95_m']) - float(r['laser_h95_m']) for r in both]),
                               'address_absdiff_median_m': med([abs(float(r['address_h95_m']) - float(r['laser_h95_m'])) for r in both]), 'c1_absdiff_median_m': med([abs(float(r['c1_h95_m']) - float(r['laser_h95_m'])) for r in both]),
                               'address_minus_c1_median_m': med([float(r['address_h95_m']) - float(r['c1_h95_m']) for r in both])}
    out['heights_from_csv'] = Cc
except FileNotFoundError:
    out['heights_from_csv'] = 'heights.csv not present yet'
print(json.dumps(out, indent=1))
if a.out: json.dump({**out, 'rederived_heads': B}, open(a.out, 'w'), indent=1)
