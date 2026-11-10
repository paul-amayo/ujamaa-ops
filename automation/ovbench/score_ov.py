#!/usr/bin/env python3
"""score_ov.py — score an open-vocabulary method's relevancy maps with the HiGH protocol (ovbench, 2026-11-10).
Input: <rel-dir>/<frame>.npz with S [H, W, Q] (the method's own relevancy, in (0, 1)) and queries [Q], one per verdict frame of the
cell (<identity>/frames.json). Heads are built exactly as argmax_eval.py builds them from the trees_only supervision:
  CAT   "tree" = every labelled pixel (ALL), "orange" = fruit pixels (ids >= 10000) — E3's category heads
  TREE  "tree <id>" = the tree's pixels + its fruit (scene-graph id from marker_hierarchy.json)
  ROW   "row <row_id>" = the union of its member trees (+ fruit)
  RIGHT "the tree on the right" = the TREE head (>= 0.2 % of the frame) whose supervised centroid lies furthest right
Per head: IoU at the method's standard threshold (relevancy >= 0.5), the oracle = best IoU over THR (the 57-value grid of
protocol_table.py / argmax_eval.py), and for TREE / ROW also argmax among the cell's addresses of that level gated by relevancy >= 0.5.
Medians over heads >= 0.2 % of the frame (protocol_table --min-frac 0.002). HiGH's lines for the same frames (protocol_v3 /
protocol_cat_v3 and the void-free v3vf / cat_v3vf when present) are summarised alongside; GT pixel counts are cross-checked.
  score_ov.py --write-queries q.json --identity <id> --sup <dir> --hier <json>          (cell query list)
  score_ov.py --rel-dir <dir> --identity <id> --sup <dir> --hier <json> --method lerf --out <jsonl> --summary <json>"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
THR = np.concatenate([np.linspace(0.4, 0.95, 45), np.linspace(0.955, 0.999, 12)]); FRUIT, UNLAB, MIN_FRAC = 10000, 65535, 0.002
ap = argparse.ArgumentParser()
ap.add_argument('--identity', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--hier', required=True)
ap.add_argument('--write-queries', default=''); ap.add_argument('--rel-dir', default=''); ap.add_argument('--method', default='')
ap.add_argument('--out', default=''); ap.add_argument('--summary', default=''); ap.add_argument('--cell', default='')
a = ap.parse_args()
N = Path(a.identity); FJ = json.load(open(N / 'frames.json')); frames = FJ['frames']; info = FJ['info']
Hj = json.load(open(a.hier)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tree_of_fruit = {f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
gt_of = lambda fr: np.array(Image.open(Path(a.sup) / fr), np.uint16)
if a.write_queries:
    meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta']))
    trees = {int(k) for k in meta['word_table'] if int(k) < FRUIT}
    for fr in frames: trees |= {int(u) for u in np.unique(gt_of(fr)) if u < FRUIT}
    rows = {int(k) for k in meta.get('row_words', {})} | {row_of[t] for t in trees if t in row_of}
    q = ['tree', 'orange', 'the tree on the right'] + [f'tree {t}' for t in sorted(trees)] + [f'row {r}' for r in sorted(rows)]
    json.dump({'queries': q, 'trees': sorted(trees), 'rows': sorted(rows)}, open(a.write_queries, 'w')); print(f'[queries] {len(q)}: {len(trees)} trees, {len(rows)} rows -> {a.write_queries}'); sys.exit()

def prf(pred, gt):
    i = np.logical_and(pred, gt).sum(); u = np.logical_or(pred, gt).sum(); return float(i / max(u, 1)), float(i / max(pred.sum(), 1)), float(i / max(gt.sum(), 1))
def sweep(heat, gt):
    return [float(np.logical_and(heat >= t, gt).sum() / max(np.logical_or(heat >= t, gt).sum(), 1)) for t in THR]
lines = []
for fr in frames:
    p = Path(a.rel_dir) / f'{fr[:-4]}.npz'
    if not p.exists(): print(f'[score] {fr}: no relevancy map — skipped', flush=True); continue
    z = np.load(p); S = z['S'].astype(np.float32); Q = [str(x) for x in z['queries']]; col = {q: i for i, q in enumerate(Q)}; H, W, _ = S.shape
    g = np.array(Image.fromarray(gt_of(fr)).resize((W, H), Image.NEAREST)); P = H * W
    uid = sorted(int(u) for u in np.unique(g) if u < UNLAB); trees = [u for u in uid if u < FRUIT]; fruits = [u for u in uid if u >= FRUIT]
    heads = [('CAT', 'tree', 'tree', g < UNLAB), ('CAT', 'orange', 'orange', (g >= FRUIT) & (g < UNLAB))]
    tree_gt = {}
    for t in trees:
        m = g == t
        for f in fruits:
            if tree_of_fruit.get(f - FRUIT) == t: m = m | (g == f)
        tree_gt[t] = m; heads.append(('TREE', t, f'tree {t}', m))
    for r in sorted({row_of[t] for t in trees if t in row_of}):
        m = np.zeros_like(g, dtype=bool)
        for t in trees:
            if row_of.get(t) == r: m |= tree_gt[t]
        heads.append(('ROW', r, f'row {r}', m))
    big = [(t, m) for t, m in tree_gt.items() if m.sum() / P >= MIN_FRAC]
    if big:
        xs = np.arange(W)[None, :]; t_r, m_r = max(big, key=lambda tm: float((tm[1] * xs).sum() / tm[1].sum()))
        heads.append(('RIGHT', t_r, 'the tree on the right', m_r))
    T_cols = [col[f'tree {t}'] for t in sorted(int(q[5:]) for q in Q if q.startswith('tree ') and q[5:].isdigit())]
    R_cols = [col[f'row {r}'] for r in sorted(int(q[4:]) for q in Q if q.startswith('row ') and q[4:].isdigit())]
    lab_t = np.array(T_cols)[S[..., T_cols].argmax(-1)]; lab_r = np.array(R_cols)[S[..., R_cols].argmax(-1)] if R_cols else None
    split = info[fr]['split']
    for kind, hid, q, gt in heads:
        if gt.sum() == 0: continue
        heat = S[..., col[q]]; cv = sweep(heat, gt); b = int(np.argmax(cv))
        rec = {'method': a.method, 'cell': a.cell, 'frame': fr, 'split': split, 'kind': kind, 'id': hid, 'query': q, 'gt_px': int(gt.sum()), 'gt_frac': float(gt.sum() / P),
               'rel05': dict(zip(('iou', 'prec', 'rec'), prf(heat >= 0.5, gt))), 'oracle': {'thr': float(THR[b]) if cv[b] > 0 else None, 'iou': cv[b]}, 'max_rel': float(heat.max()), 'curve_thr': [round(x, 4) for x in cv]}
        if kind in ('TREE', 'RIGHT'): rec['argmax05'] = dict(zip(('iou', 'prec', 'rec'), prf((lab_t == col[f'tree {hid}']) & (S[..., col[f'tree {hid}']] >= 0.5), gt)))
        if kind == 'ROW' and lab_r is not None: rec['argmax05'] = dict(zip(('iou', 'prec', 'rec'), prf((lab_r == col[q]) & (heat >= 0.5), gt)))
        lines.append(rec)
    print(f'[score] {fr} ({split}): ' + ' | '.join(f'{l["query"]} {l["rel05"]["iou"]:.3f}/{l["oracle"]["iou"]:.3f}' for l in lines if l['frame'] == fr), flush=True)
with open(a.out, 'w') as f:
    for l in lines: f.write(json.dumps(l) + '\n')
# ---- summary: method medians + HiGH medians on the same frames (heads >= 0.2 % of the frame)
def med(v): return float(np.median(v)) if len(v) else None
def summ(L, key):
    return {'n': len(L), 'median_rel05' if key == 'rel05' else 'median_' + key: med([l[key]['iou'] for l in L])} if L else {'n': 0}
S_ = {}
for kind, q, name in (('CAT', 'tree', 'cat_tree'), ('CAT', 'orange', 'cat_orange'), ('TREE', None, 'tree_address'), ('RIGHT', None, 'tree_on_the_right'), ('ROW', None, 'row_address')):
    for split in ('eval', 'train'):
        L0 = [l for l in lines if l['kind'] == kind and l['split'] == split and (q is None or l['query'] == q)]; L = [l for l in L0 if l['gt_frac'] >= MIN_FRAC]
        S_[name + '/' + split] = {'n_heads': len(L), 'median_iou_rel05': med([l['rel05']['iou'] for l in L]), 'median_iou_oracle': med([l['oracle']['iou'] for l in L]),
                 'median_iou_argmax05': med([l['argmax05']['iou'] for l in L if 'argmax05' in l]) if kind in ('TREE', 'ROW', 'RIGHT') else None,
                 'n_heads_unfiltered': len(L0), 'median_iou_rel05_unfiltered': med([l['rel05']['iou'] for l in L0]), 'median_iou_oracle_unfiltered': med([l['oracle']['iou'] for l in L0])}
high = {}; checks = []
for tag, fn in (('v3', 'protocol_v3.jsonl'), ('v3vf', 'protocol_v3vf.jsonl'), ('cat_v3', 'protocol_cat_v3.jsonl'), ('cat_v3vf', 'protocol_cat_v3vf.jsonl')):
    if not (N / fn).exists(): continue
    HL = [json.loads(x) for x in open(N / fn)]; HL = [l for l in HL if l['kind'] != 'FRAME' and l['frame'] in frames]
    gk = {(l['frame'], l['kind'], l['id']): l['gt_px'] for l in HL}
    for l in lines:   # GT cross-check: our head pixel counts must equal HiGH's for the same head
        key = (l['frame'], 'CAT', 0 if l['query'] == 'tree' else 1) if l['kind'] == 'CAT' else (l['frame'], l['kind'], l['id'])
        if key in gk and l['kind'] != 'RIGHT': checks.append(gk[key] == l['gt_px'])
    for split in ('eval', 'train'):
        if tag.startswith('cat'):
            for w in ('tree', 'orange'):
                L0 = [l for l in HL if l['split'] == split and l.get('word') == w]; L = [l for l in L0 if l['gt_frac'] >= MIN_FRAC]
                high[f'{tag}:cat_{w}/{split}'] = {'n_heads': len(L), 'median_iou_rel05': med([l['rel05']['iou'] for l in L]), 'median_iou_oracle': med([l['oracle']['iou'] for l in L]), 'median_iou_argmax': med([l['argmax']['iou'] for l in L]),
                                                  'n_heads_unfiltered': len(L0), 'median_iou_rel05_unfiltered': med([l['rel05']['iou'] for l in L0]), 'median_iou_oracle_unfiltered': med([l['oracle']['iou'] for l in L0])}
        else:
            for kind in ('TREE', 'ROW'):
                L = [l for l in HL if l['split'] == split and l['kind'] == kind and l['gt_frac'] >= MIN_FRAC]
                high[f'{tag}:{kind.lower()}_address/{split}'] = {'n_heads': len(L), 'median_iou_oracle': med([l['oracle']['iou'] for l in L]), 'median_iou_argmax': med([l['argmax']['iou'] for l in L])}
            R = [l for l in lines if l['kind'] == 'RIGHT' and l['split'] == split and l['gt_frac'] >= MIN_FRAC]; hk = {(l['frame'], l['id']): l for l in HL if l['kind'] == 'TREE'}
            Lr = [hk[(l['frame'], l['id'])] for l in R if (l['frame'], l['id']) in hk]
            high[f'{tag}:tree_on_the_right_by_address/{split}'] = {'n_heads': len(Lr), 'median_iou_oracle': med([l['oracle']['iou'] for l in Lr]), 'median_iou_argmax': med([l['argmax']['iou'] for l in Lr])}
out = {'method': a.method, 'cell': a.cell, 'identity': str(N), 'frames': len({l['frame'] for l in lines}), 'min_frac': MIN_FRAC, 'thr_grid': '57 values (protocol_table.THR)',
       'gt_check': {'heads_compared': len(checks), 'equal': int(sum(checks))}, 'method_medians': S_, 'high_medians': high}
json.dump(out, open(a.summary, 'w'), indent=1)
print(f'[score] GT cross-check vs HiGH lines: {sum(checks)}/{len(checks)} equal pixel counts', flush=True)
for k, v in S_.items():
    if k.endswith('/eval'): print(f'[score] {a.method} {k}: n={v["n_heads"]} rel>=0.5 {v["median_iou_rel05"]} oracle {v["median_iou_oracle"]} argmax05 {v["median_iou_argmax05"]}', flush=True)
for k, v in high.items():
    if k.endswith('/eval'): print(f'[score] HiGH {k}: {v}', flush=True)
