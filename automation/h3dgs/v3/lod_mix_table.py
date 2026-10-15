#!/usr/bin/env python3
"""lod_mix_table.py — table of lod_eval_ckpt.sh outputs (2026-10-14): per arm/checkpoint, gaussians, purity (share of the oranges' rendered mass from
blobs >= 50 % fruit by labelled votes; mixed share), E4 fruit under the 10 % rule and the labelled-majority rule (census-view and held-out IoU / P / R,
tree IoU), fruit-frame fruit PSNR.   system python3:  lod_mix_table.py --rows "label=<identity dir>" ... [--md out.md]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--rows', nargs='+', required=True); ap.add_argument('--md', default=''); a = ap.parse_args()
def e4(p):
    if not os.path.exists(p): return '—', '—', '—'
    H = [json.loads(l) for l in open(p)]; o = []
    for sp in ('train', 'eval'):
        F = [h for h in H if h['kind'] == 'FRUIT' and h['split'] == sp]; o.append(f"{np.median([h['argmax']['iou'] for h in F]):.2f} / {np.median([h['argmax']['prec'] for h in F]):.2f} / {np.median([h['argmax']['rec'] for h in F]):.2f}" if F else '—')
    T = [h for h in H if h['kind'] == 'TREE' and h['split'] == 'train']; return o[0], o[1], f"{np.median([h['argmax']['iou'] for h in T]):.2f}" if T else '—'
L = ['| arm @ step | gaussians | purity: mass from pure-fruit blobs / from mixed (fruit px, census views) | pure-fruit blobs | E4 10 % rule: census IoU/P/R · held-out | E4 labelled-majority: census · held-out | E4 mask-views majority: census · held-out | tree IoU | fruit PSNR census / held-out (fruit frames) |', '|---|---|---|---|---|---|---|---|---|']
for spec in a.rows:
    lab, d = spec.split('=', 1); pu = json.load(open(f'{d}/purity.json')) if os.path.exists(f'{d}/purity.json') else None; w = json.load(open(f'{d}/W.json')) if os.path.exists(f'{d}/W.json') else {}
    ps = json.load(open(f'{d}/masked_psnr_fruitframes.json')) if os.path.exists(f'{d}/masked_psnr_fruitframes.json') else None
    f1 = next((q for q in (f'{d}/protocol_e4_s01_fruit_level.jsonl', f'{d}/protocol_e4_s01.jsonl') if os.path.exists(q)), ''); f5 = next((q for q in (f'{d}/protocol_e4_s05_fruit_level.jsonl', f'{d}/protocol_e4_s05.jsonl') if os.path.exists(q)), '')
    c1, e1, t1 = e4(f1); c5, e5, t5 = e4(f5); cm, em, tm = e4(f'{d}/protocol_e4_mv_fruit_level.jsonl')
    L.append(f"| {lab} | {w.get('gaussians', '—')} | {pu['pure_share_over_all_fruit_px']:.2f} / {pu['mixed_share_over_all_fruit_px']:.2f} | {pu['blobs_pure_fruit']} | {c1} · {e1} | {c5} · {e5} | {cm} · {em} | {t1} | "
             + (f"{ps['train']['fruit']['psnr']:.2f} / {ps['eval']['fruit']['psnr']:.2f}" if ps else '—') + ' |' if pu else f'| {lab} | pending | | | | | | |')
t = '\n'.join(L); print(t)
if a.md: open(a.md, 'w').write(t + '\n')
