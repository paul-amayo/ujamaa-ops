#!/usr/bin/env python3
"""Native-vs-side-car containment table from a native_cell_test verdicts.log (UJAMAA, 2026-10-03). Per frame and head:
IoU of each model; means by kind (TREE / ROW / FRUIT) and split (train = in both censuses, eval = held out of both);
wins / ties / losses at |d| <= --tie. --canonical <log> --canonical-tag <tag>: check that the npz-scored side-car lines
reproduce the verdicts the canonical containment_eval recorded for the same run (validates the --features-npz path)."""
import argparse, json, re
from collections import defaultdict
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--verdicts', required=True); ap.add_argument('--frames', required=True)
ap.add_argument('--models', nargs=2, default=['native', 'sidecar']); ap.add_argument('--tie', type=float, default=0.02)
ap.add_argument('--canonical', default=''); ap.add_argument('--canonical-tag', default='')
a = ap.parse_args()
info = json.load(open(a.frames))['info']
pat = re.compile(r'^\[(\S+) (\S+)\] (TREE|ROW|FRUIT of) +(\d+) +"([^"]*)": thr (\S+) IoU (\S+) prec (\S+) rec (\S+)')
gate = re.compile(r'^\[(\S+) (\S+)\] \[norm gate\] ([\d.]+)%')
R = defaultdict(dict); void = defaultdict(dict)
for line in open(a.verdicts, errors='replace'):
    m = pat.match(line)
    if m:
        model, fr, kind, hid, word, thr, iou, prec, rec = m.groups()
        R[(fr, kind.replace(' of', ''), int(hid), word)][model] = (float(iou), float(prec), float(rec), thr)
    g = gate.match(line)
    if g: void[g.group(2)][g.group(1)] = float(g.group(3))
m0, m1 = a.models
print(f'| frame | split | head | {m0} IoU | {m1} IoU | d |'); print('|---|---|---|---|---|---|')
agg = defaultdict(lambda: defaultdict(list)); wtl = defaultdict(lambda: [0, 0, 0])
for key in sorted(R):
    fr, kind, hid, word = key; v = R[key]
    if m0 not in v or m1 not in v: continue
    x, y = v[m0][0], v[m1][0]; d = x - y; sp = info.get(fr, {}).get('split', '?')
    print(f'| {fr} | {sp} | {kind} {hid} "{word}" | {x:.3f} | {y:.3f} | {d:+.3f} |')
    agg[(kind, sp)][m0].append(x); agg[(kind, sp)][m1].append(y)
    wtl[(kind, sp)][0 if d > a.tie else (2 if d < -a.tie else 1)] += 1
print(); print(f'| kind | split | n | {m0} mean | {m1} mean | {m0} wins / ties / losses |'); print('|---|---|---|---|---|---|')
for k in sorted(agg):
    n = len(agg[k][m0]); w, t, l = wtl[k]
    print(f'| {k[0]} | {k[1]} | {n} | {np.mean(agg[k][m0]):.3f} | {np.mean(agg[k][m1]):.3f} | {w} / {t} / {l} |')
print(); print('void (rendered ||f|| < 0.5, % of frame): ' + ', '.join(f"{fr} {void[fr].get(m0, float('nan')):.1f}/{void[fr].get(m1, float('nan')):.1f}" for fr in sorted(void)))
if a.canonical:
    canon = {}
    cpat = re.compile(r'^\[' + re.escape(a.canonical_tag) + r' (\S+)\] (TREE|ROW|FRUIT of) +(\d+) +"([^"]*)": thr (\S+) IoU (\S+)')
    for line in open(a.canonical, errors='replace'):
        m = cpat.match(line)
        if m: canon[(m.group(1), m.group(2).replace(' of', ''), int(m.group(3)), m.group(4))] = (float(m.group(6)), m.group(5))
    hits = [(k, canon[k], R[k][m1]) for k in canon if k in R and m1 in R[k]]
    exact = sum(1 for k, c, v in hits if abs(c[0] - v[0]) < 1e-3 and c[1] == v[3])
    print(f'\ncanonical check ({a.canonical_tag}): {exact}/{len(hits)} side-car verdicts reproduced exactly (IoU and thr)')
    for k, c, v in hits:
        if not (abs(c[0] - v[0]) < 1e-3 and c[1] == v[3]): print(f'  differs: {k}: canonical IoU {c[0]:.3f} thr {c[1]} vs npz IoU {v[0]:.3f} thr {v[3]}')
