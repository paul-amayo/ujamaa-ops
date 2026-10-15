#!/usr/bin/env python3
"""lod_table.py — before/after table for the mask-size LoD runs of one cell (2026-10-13): the stock v3 run vs each identity_lod_<tag>
(or identity_<name>) dir — gaussians, masked PSNR by object size (census views with the grid / held-out raw; whole frame, all masks, fruit,
< 0.2 % stratum, COCO small/medium), fruit-frame fruit PSNR, the E4 fruit scores (level rule: argmax IoU median + instance recall, census and
held-out), the tree argmax IoU (must not regress) and the per-orange counts from the orange figure json when present.
system python3:  lod_table.py --ws <workspace> --stock-identity <v3 identity dir> [--runs v1 v2 ...] [--md out.md] [--json out.json]"""
import argparse, json, os, glob
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--ws', required=True); ap.add_argument('--stock-identity', required=True); ap.add_argument('--runs', nargs='*', default=None)
ap.add_argument('--md', default=''); ap.add_argument('--json', default=''); a = ap.parse_args()
med = lambda v: float(np.median(v)) if len(v) else float('nan')
def heads(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def e4(path):
    H = [h for h in heads(path) if h['kind'] != 'FRAME']; d = {}
    for sp, nm in (('train', 'census'), ('eval', 'held-out')):
        F = [h for h in H if h['kind'] == 'FRUIT' and h['split'] == sp]; T = [h for h in H if h['kind'] == 'TREE' and h['split'] == sp]
        d[nm] = {'fruit_heads': len(F), 'fruit_argmax_med': med([h['argmax']['iou'] for h in F]), 'fruit_oracle_med': med([h['oracle']['iou'] for h in F]),
                 'fruit_inst_recall_argmax': float(np.mean([h['argmax']['rec'] > 0 for h in F])) if F else float('nan'), 'tree_argmax_med': med([h['argmax']['iou'] for h in T])}
    return d
def psnr(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); g = lambda s, k: d[s][k]['psnr'] if k in d[s] and d[s][k]['px'] else float('nan')
    return {s: {k: g(s, k) for k in ('frame', 'all_masks', 'fruit', '>=3%', '1-3%', '0.2-1%', '<0.2%', 'small<32^2', 'medium', 'large>96^2')} | {'frames': d[s]['frames']} for s in ('train', 'eval')}
def oranges(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); r = d['runs'].get('after') or d['runs'].get('before'); return {'frame': d['frame'], 'n_on_orange': [o['n_on_orange'] for o in r['oranges']], 'median_cm': [o['median_cm'] for o in r['oranges']], 'psnr_mask_db': [o['psnr_mask_db'] for o in r['oranges']], 'gaussians': r['gaussians']}
rows = {}
S = a.stock_identity; rows['stock'] = {'psnr': psnr(f'{S}/masked_psnr_stock.json'), 'psnr_fruitframes': psnr(f'{S}/masked_psnr_stock_fruitframes.json'), 'e4_level': e4(f'{S}/protocol_v3_fruit_level.jsonl'), 'e4_mixed': e4(f'{S}/protocol_v3_fruit.jsonl'),
                 'oranges': oranges(f'{S}/figs/lod/lod_oranges_before.json'), 'gaussians': json.load(open(f'{S}/features_B.bin.json'))['n'] if os.path.exists(f'{S}/features_B.bin.json') else None}
runs = a.runs if a.runs is not None else sorted(os.path.basename(p)[len('identity_'):] for p in glob.glob(f'{a.ws}/identity_*lod*'))
for r in runs:
    I = f'{a.ws}/identity_{r}' if os.path.isdir(f'{a.ws}/identity_{r}') else f'{a.ws}/identity_lod_{r}'; tag = os.path.basename(I)[len('identity_'):]
    short = tag[4:] if tag.startswith('lod_') else tag   # figure jsons are named by the short variant tag (v1, v2, px1024_lod_v2)
    fig = glob.glob(f'{S}/figs/lod/lod_oranges_{r}.json') + glob.glob(f'{S}/figs/lod/lod_oranges_{tag}.json') + glob.glob(f'{S}/figs/lod/lod_oranges_{short}.json')
    ptag = tag.split('_', 1)[1] if tag.startswith('px') else tag   # cells name their protocols lod<TAG> (identity_px1024_lod_v2 -> protocol_lod_v2_*)
    rows[tag] = {'psnr': psnr(f'{I}/masked_psnr.json'), 'psnr_fruitframes': psnr(f'{I}/masked_psnr_fruitframes.json'), 'e4_level': e4(f'{I}/protocol_{ptag}_fruit_level.jsonl'), 'e4_mixed': e4(f'{I}/protocol_{ptag}_fruit_mixed.jsonl'),
                 'oranges': oranges(fig[0]) if fig else None, 'gaussians': json.load(open(f'{I}/features_B.bin.json'))['n'] if os.path.exists(f'{I}/features_B.bin.json') else None}
f = lambda x: '—' if x is None or (isinstance(x, float) and np.isnan(x)) else (f'{x:.2f}' if isinstance(x, float) else str(x))
L = ['| run | gaussians | census: frame / masks / fruit / <0.2% / COCO-S / COCO-M | held-out: frame / masks / fruit / <0.2% / COCO-S / COCO-M | fruit frames: fruit (census / held-out) | E4 level census: IoU, inst-recall | E4 level held-out: IoU, inst-recall | tree IoU census / held-out | oranges on fig frame: n (median cm) |', '|---|---|---|---|---|---|---|---|---|']
for name, r in rows.items():
    p, q, e = r['psnr'], r['psnr_fruitframes'], r['e4_level']; o = r['oranges']
    c = lambda d, s: ' / '.join(f(d[s][k]) for k in ('frame', 'all_masks', 'fruit', '<0.2%', 'small<32^2', 'medium')) if d else '—'
    L.append(f"| {name} | {f(r['gaussians'])} | {c(p, 'train')} | {c(p, 'eval')} | {f(q['train']['fruit']) if q else '—'} / {f(q['eval']['fruit']) if q else '—'} | "
             f"{f(e['census']['fruit_argmax_med'])}, {f(e['census']['fruit_inst_recall_argmax'])} (n={e['census']['fruit_heads']}) | {f(e['held-out']['fruit_argmax_med'])}, {f(e['held-out']['fruit_inst_recall_argmax'])} (n={e['held-out']['fruit_heads']}) | "
             f"{f(e['census']['tree_argmax_med'])} / {f(e['held-out']['tree_argmax_med'])} | {', '.join(f'{n} ({m:.1f})' for n, m in zip(o['n_on_orange'], o['median_cm'])) if o else '—'} |")
txt = '\n'.join(L); print(txt)
if a.md: open(a.md, 'w').write(txt + '\n')
if a.json: json.dump(rows, open(a.json, 'w'), indent=1)
