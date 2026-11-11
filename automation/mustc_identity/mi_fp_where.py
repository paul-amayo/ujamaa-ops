#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3), A100 session 11-11. POST HOC diagnostic, not a rule and not part of the design: for each head of a view,
where do its false-positive pixels (lit with the head's address, outside the head's reference) fall? Tallied against the view's SAM 3 masks
and their association outcome (masks.csv): another labelled plant, an ambiguous mask (with the census ids it holds), an off-row-only mask, a
mask with no census point, or no mask at all.
nerf_new pixi env:  mi_fp_where.py --dir <stop dir> --view nikon_5 [--tag tangent_vf] [--out fp_where_<view>.json]"""
import argparse, csv, json
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--view', required=True); ap.add_argument('--tag', default='tangent_vf'); ap.add_argument('--out', default='')
ap.add_argument('--hier', default='/mnt/shared/must-c_bulk/work/identity_feas/identity/marker_hierarchy.json'); a = ap.parse_args(); D = a.dir
C = json.load(open(f'{D}/cameras.json')); name = next(n for n, c in C['frames'].items() if c['label'] == a.view)
pr = np.load(f'{D}/pred_{a.tag}/{name}.npz'); tree, row = pr['tree'], pr['row']; h, w = tree.shape
gt = np.array(Image.fromarray(np.array(Image.open(f'{D}/cell/supervision/trees_only/{name}'), np.uint16)).resize((w, h), Image.NEAREST)).astype(np.int64)
sam = np.array(Image.fromarray(np.array(Image.open(f'{D}/masks/{a.view}.png'), np.uint16)).resize((w, h), Image.NEAREST)).astype(np.int64)
oc = {int(r['idx']): (r['outcome'], r['ids']) for r in csv.DictReader(open(f'{D}/cell/masks.csv')) if r['view'] == a.view}
row_of = {o['id']: o['row_id'] for o in json.load(open(a.hier))['objects']}
def where(fp):
    out = {}
    lab = gt[fp]; s = sam[fp]
    for l_, s_ in zip(lab.tolist(), s.tolist()):
        if l_ != 65535: k = f'labelled plant {l_} (row {row_of.get(l_)})'
        elif s_ == 0: k = 'no SAM mask'
        else:
            o, ids = oc.get(s_, ('?', '')); k = f'{o} mask' + (f' [census ids {ids}]' if ids else '')
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
res = {'view': a.view, 'note': 'POST HOC diagnostic, not a rule', 'heads': []}
ids = [int(u) for u in np.unique(gt) if u != 65535]
for kind, i, ref, pred in [('TREE', i, gt == i, tree == i) for i in ids] + [('ROW', r, np.isin(gt, [j for j in ids if row_of.get(j) == r]), row == r) for r in sorted({row_of[j] for j in ids})]:
    fp = pred & ~ref; res['heads'].append({'kind': kind, 'id': i, 'pred_px': int(pred.sum()), 'ref_px': int(ref.sum()), 'fp_px': int(fp.sum()), 'fp_where': where(fp)})
    print(f"[fp] {a.view} {kind} {i}: pred {int(pred.sum())} px, ref {int(ref.sum())} px, false positive {int(fp.sum())} px -> {res['heads'][-1]['fp_where']}")
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
