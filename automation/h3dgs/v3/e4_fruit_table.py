#!/usr/bin/env python3
"""e4_fruit_table.py — the fruit level on fruit-picked frames (paper E4, 2026-10-13): per survey, FRUIT heads on the held-out and census frames of
protocol_v3_fruit.jsonl — heads, pixel IoU (oracle / argmax), precision / recall medians, INSTANCE RECALL (fraction of GT fruit heads with any lit
pixel under the argmax rule, and under the oracle threshold), by size stratum; plus the tree numbers on the same frames for reference.
system python3:  e4_fruit_table.py [--inventory ...] [--tag v3_fruit] [--md out.md] [--json out.json]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--inventory', default='/home/paperspace/logs/identity_inventory_v2.jsonl'); ap.add_argument('--tag', default='v3_fruit'); ap.add_argument('--md', default=''); ap.add_argument('--json', default=''); a = ap.parse_args()
cells = [json.loads(l) for l in open(a.inventory)]; H = []
for c in cells:
    p = f"{c['identity']}/protocol_{a.tag}.jsonl"
    if os.path.exists(p):
        for r in map(json.loads, open(p)):
            if r['kind'] != 'FRAME': r['survey'] = c['survey']; r['cell'] = c['identity']; H.append(r)
med = lambda v: float(np.median(v)) if len(v) else float('nan'); out = {}; L = [f'# E4 fruit on fruit-picked frames ({a.tag})', '', '| survey | split | fruit heads | cells | oracle IoU | argmax IoU | argmax prec | argmax rec | instance recall (argmax) | instance recall (oracle) | <0.2 % n: argmax | 0.2-1 % n: argmax | tree IoU argmax (same frames) |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for sv in sorted({h['survey'] for h in H}):
    for sp in ('train', 'eval'):
        F = [h for h in H if h['survey'] == sv and h['kind'] == 'FRUIT' and h['split'] == sp]; T = [h for h in H if h['survey'] == sv and h['kind'] == 'TREE' and h['split'] == sp]
        if not F: continue
        d = {'n': len(F), 'cells': len({h['cell'] for h in F}), 'oracle_med': med([h['oracle']['iou'] for h in F]), 'argmax_med': med([h['argmax']['iou'] for h in F]), 'argmax_prec_med': med([h['argmax']['prec'] for h in F]), 'argmax_rec_med': med([h['argmax']['rec'] for h in F]),
             'inst_recall_argmax': float(np.mean([h['argmax']['rec'] > 0 for h in F])), 'inst_recall_oracle': float(np.mean([h['oracle']['rec'] > 0 for h in F])), 'tree_argmax_med': med([h['argmax']['iou'] for h in T]),
             'strata': {lab: {'n': len(v), 'argmax_med': med([h['argmax']['iou'] for h in v])} for lo, hi, lab in ((0, 0.002, '<0.2%'), (0.002, 0.01, '0.2-1%'), (0.01, 1.01, '>=1%')) for v in [[h for h in F if lo <= h['gt_frac'] < hi]]}}
        out.setdefault(sv, {})['held-out' if sp == 'eval' else 'census'] = d; s = d['strata']
        L.append(f"| {sv} | {'held-out' if sp == 'eval' else 'census'} | {d['n']} | {d['cells']} | {d['oracle_med']:.3f} | {d['argmax_med']:.3f} | {d['argmax_prec_med']:.3f} | {d['argmax_rec_med']:.3f} | {d['inst_recall_argmax']:.2f} | {d['inst_recall_oracle']:.2f} | {s['<0.2%']['n']}: {s['<0.2%']['argmax_med']:.3f} | {s['0.2-1%']['n']}: {s['0.2-1%']['argmax_med']:.3f} | {d['tree_argmax_med']:.3f} |")
txt = '\n'.join(L); print(txt)
if a.md: open(a.md, 'w').write(txt + '\n')
if a.json: json.dump(out, open(a.json, 'w'), indent=1)
