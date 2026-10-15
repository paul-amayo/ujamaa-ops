#!/usr/bin/env python3
"""protocol_table.py — aggregate argmax_eval.py outputs (<identity>/protocol_<tag>.jsonl) into the paper's E1 tables (2026-10-13).
Per survey x level x split: heads, oracle (best threshold per head = the upper bound), argmax (threshold-free), fixed-cell (ONE
presence threshold per level fitted on the cell's census heads under the argmax rule, applied to its held-out heads), fixed-survey
(the survey's pooled census heads), argmax precision / recall, held-out size strata (GT fraction of frame >= 3 %, 1-3 %, 0.2-1 %,
< 0.2 %), nesting consistency and void fraction per frame. Cross-survey transfer: survey X's threshold on survey Y's held-out heads.
system python3:  protocol_table.py [--inventory /home/paperspace/logs/identity_inventory.jsonl] [--tag v3] [--md out.md] [--json out.json]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--inventory', default='/home/paperspace/logs/identity_inventory.jsonl'); ap.add_argument('--tag', default='v3')
ap.add_argument('--md', default=''); ap.add_argument('--json', default=''); ap.add_argument('--surveys', default='')
ap.add_argument('--compare', default='', help='comma-separated tags (ablation arms) to put side by side on the heads they all scored, e.g. v3,hyper_own,flat96,euclid'); a = ap.parse_args()
THR = np.concatenate([np.linspace(0.4, 0.95, 45), np.linspace(0.955, 0.999, 12)]); STRATA = ((0.03, 1.01, '>=3%'), (0.01, 0.03, '1-3%'), (0.002, 0.01, '0.2-1%'), (0, 0.002, '<0.2%'))
cells = [json.loads(l) for l in open(a.inventory)]; heads, frames = [], []
for c in cells:
    p = f"{c['identity']}/protocol_{a.tag}.jsonl"
    if not os.path.exists(p): continue
    for r in map(json.loads, open(p)):
        r['survey'] = c['survey']; r['cell'] = c['identity']; (frames if r['kind'] == 'FRAME' else heads).append(r)
surveys = [s for s in dict.fromkeys(c['survey'] for c in cells) if any(h['survey'] == s for h in heads)]
if a.surveys: surveys = [s for s in surveys if s in a.surveys.split(',')]
med = lambda v: float(np.median(v)) if len(v) else float('nan'); mean = lambda v: float(np.mean(v)) if len(v) else float('nan')
def fit(hs):   # threshold index maximising the mean argmax+threshold IoU over a set of census heads
    return int(np.argmax(np.mean([h['curve_argmax_thr'] for h in hs], 0))) if hs else None
thr_s = {}; out = {'tag': a.tag, 'cells': sum(1 for c in cells if os.path.exists(f"{c['identity']}/protocol_{a.tag}.jsonl")), 'surveys': {}, 'transfer': {}}
rows = []
for s in surveys:
    H = [h for h in heads if h['survey'] == s]; F = [f for f in frames if f['survey'] == s]; out['surveys'][s] = {'cells': len({h['cell'] for h in H}), 'frames': len(F),
        'nesting_mean': mean([f['nesting'] for f in F if f['nesting'] == f['nesting']]), 'void_mean': mean([f['void_frac'] for f in F])}
    for kind in ('TREE', 'ROW', 'FRUIT'):
        K = [h for h in H if h['kind'] == kind]
        if not K: continue
        tr = [h for h in K if h['split'] == 'train']; ev = [h for h in K if h['split'] == 'eval']; i_s = fit(tr); thr_s[(s, kind)] = i_s
        fixed_cell = []
        for cell in {h['cell'] for h in ev}:
            i_c = fit([h for h in tr if h['cell'] == cell])
            if i_c is None: i_c = i_s
            if i_c is not None: fixed_cell += [h['curve_argmax_thr'][i_c] for h in ev if h['cell'] == cell]
        fixed_s = [h['curve_argmax_thr'][i_s] for h in ev] if i_s is not None else []
        d = {}
        for name, hs in (('train', tr), ('eval', ev)):
            d[name] = {'n': len(hs), 'oracle_med': med([h['oracle']['iou'] for h in hs]), 'oracle_mean': mean([h['oracle']['iou'] for h in hs]),
                       'argmax_med': med([h['argmax']['iou'] for h in hs]), 'argmax_mean': mean([h['argmax']['iou'] for h in hs]),
                       'argmax_prec_med': med([h['argmax']['prec'] for h in hs]), 'argmax_rec_med': med([h['argmax']['rec'] for h in hs])}
        d['eval'].update({'fixed_cell_med': med(fixed_cell), 'fixed_cell_mean': mean(fixed_cell), 'fixed_survey_med': med(fixed_s), 'fixed_survey_mean': mean(fixed_s),
                          'thr_survey': float(THR[i_s]) if i_s is not None else None,
                          'strata': {lab: {'n': len(v), 'argmax_med': med([h['argmax']['iou'] for h in v]), 'oracle_med': med([h['oracle']['iou'] for h in v])}
                                     for lo, hi, lab in STRATA for v in [[h for h in ev if lo <= h['gt_frac'] < hi]]}})
        if kind == 'TREE': d['eval']['sibling_confusion_med'] = med([h.get('sibling_confusion', np.nan) for h in ev]); d['eval']['nest_pred_mean'] = mean([h['nest_pred'] for h in ev if 'nest_pred' in h])
        out['surveys'][s][kind] = d; rows.append((s, kind, d))
for s in surveys:   # cross-survey transfer of the TREE threshold
    for t in surveys:
        i = thr_s.get((s, 'TREE')); ev = [h for h in heads if h['survey'] == t and h['kind'] == 'TREE' and h['split'] == 'eval']
        if i is not None and ev: out['transfer'].setdefault(s, {})[t] = med([h['curve_argmax_thr'][i] for h in ev])
L = [f"# E1 protocol table ({a.tag}) — {out['cells']} cells, {len(heads)} heads, {len(frames)} frames", '',
     '| survey | level | census n | oracle | argmax | held-out n | oracle | argmax (prec / rec) | fixed cell | fixed survey (thr) | >=3% n: argmax | 1-3% | 0.2-1% | <0.2% |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for s, kind, d in rows:
    e, t = d['eval'], d['train']; st = e['strata']
    L.append(f"| {s} | {kind} | {t['n']} | {t['oracle_med']:.3f} | {t['argmax_med']:.3f} | {e['n']} | {e['oracle_med']:.3f} | {e['argmax_med']:.3f} ({e['argmax_prec_med']:.2f} / {e['argmax_rec_med']:.2f}) | {e['fixed_cell_med']:.3f} | {e['fixed_survey_med']:.3f} ({e['thr_survey'] if e['thr_survey'] is None else round(e['thr_survey'], 2)}) | "
             + ' | '.join(f"{st[l]['n']}: {st[l]['argmax_med']:.3f}" for _, _, l in STRATA) + ' |')
L += ['', '| survey | cells | frames | nesting (mean) | void (mean) | TREE sibling confusion (held-out med) |', '|---|---|---|---|---|---|']
for s in surveys:
    v = out['surveys'][s]; L.append(f"| {s} | {v['cells']} | {v['frames']} | {v['nesting_mean']:.3f} | {v['void_mean']:.3f} | {v.get('TREE', {}).get('eval', {}).get('sibling_confusion_med', float('nan')):.3f} |")
if out['transfer']:
    L += ['', 'TREE held-out argmax IoU (median) with the row survey\'s fixed threshold applied to the column survey:', '', '| thr from \\ applied to | ' + ' | '.join(surveys) + ' |', '|---|' + '---|' * len(surveys)]
    for s in surveys: L.append(f'| {s} | ' + ' | '.join(f"{out['transfer'].get(s, {}).get(t, float('nan')):.3f}" for t in surveys) + ' |')
if a.compare:   # ablation arms side by side: held-out argmax IoU medians per survey x level over the heads every arm scored (+ nesting per frame)
    tags = a.compare.split(','); byk = {}
    for c in cells:
        for t in tags:
            p = f"{c['identity']}/protocol_{t}.jsonl"
            if not os.path.exists(p): continue
            for r in map(json.loads, open(p)):
                key = (c['survey'], c['identity'], r['frame'], r['kind'], r.get('id'), r.get('word')); byk.setdefault(key, {})[t] = r
    out['arms'] = {'tags': tags, 'rows': []}
    L += ['', f'## Ablation arms ({a.compare}) — held-out argmax IoU median (n heads scored by every arm); nesting = mean over frames', '', '| survey | level | n | ' + ' | '.join(tags) + ' |', '|---|---|---|' + '---|' * len(tags)]
    for s_ in surveys:
        cat_words = sorted({k[5] for k in byk if k[0] == s_ and k[3] == 'CAT'})
        for kind in ('TREE', 'ROW', 'FRUIT') + tuple(f'CAT:{w}' for w in cat_words) + ('FRAME',):
            if kind.startswith('CAT:'): rows_ = [v for k, v in byk.items() if k[0] == s_ and k[3] == 'CAT' and k[5] == kind[4:] and all(t in v for t in tags) and v[tags[0]]['split'] == 'eval']
            else: rows_ = [v for k, v in byk.items() if k[0] == s_ and k[3] == kind and all(t in v for t in tags) and (kind == 'FRAME' or v[tags[0]]['split'] == 'eval')]
            if not rows_: continue
            if kind == 'FRAME':
                vals = {t: mean([v[t]['nesting'] for v in rows_ if v[t]['nesting'] == v[t]['nesting']]) for t in tags}; L.append(f"| {s_} | nesting | {len(rows_)} | " + ' | '.join(f"{vals[t]:.3f}" for t in tags) + ' |')
                out['arms']['rows'].append({'survey': s_, 'level': 'nesting', 'n': len(rows_), 'values': vals})
            else:
                vals = {t: med([v[t]['argmax']['iou'] for v in rows_]) for t in tags}; L.append(f"| {s_} | {kind} | {len(rows_)} | " + ' | '.join(f"{vals[t]:.3f}" for t in tags) + ' |')
                near = [v for v in rows_ if v[tags[0]].get('gt_frac', 0) >= 0.03]
                row = {'survey': s_, 'level': kind, 'n': len(rows_), 'values': vals, 'near_n': len(near), 'near_values': {t: med([v[t]['argmax']['iou'] for v in near]) for t in tags}}
                if kind.startswith('CAT:'): row['oracle_values'] = {t: med([v[t]['oracle']['iou'] for v in rows_]) for t in tags}; row['rel05_values'] = {t: med([v[t]['rel05']['iou'] for v in rows_ if 'rel05' in v[t]]) for t in tags}
                out['arms']['rows'].append(row)
        ru = [v for k, v in byk.items() if k[0] == s_ and k[3] == 'ROW' and all(t in v for t in tags) and v[tags[0]]['split'] == 'eval' and all('argmax_union' in v[t] for t in tags)]
        if ru:
            vals = {t: med([v[t]['argmax_union']['iou'] for v in ru]) for t in tags}; L.append(f"| {s_} | ROW (union of member trees) | {len(ru)} | " + ' | '.join(f"{vals[t]:.3f}" for t in tags) + ' |')
            out['arms']['rows'].append({'survey': s_, 'level': 'ROW_union', 'n': len(ru), 'values': vals})
txt = '\n'.join(L); print(txt)
if a.md: open(a.md, 'w').write(txt + '\n')
if a.json: os.makedirs(os.path.dirname(a.json) or '.', exist_ok=True); json.dump(out, open(a.json, 'w'), indent=1)
