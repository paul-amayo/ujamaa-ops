#!/usr/bin/env python3
"""Pick verdict frames for a native-vs-side-car comparison (UJAMAA, 2026-10-03): the --must frames (recorded verdict
frames) + the most-labelled frame (>= 2 trees preferred) of each of --n-train equal segments of the side-car TRAIN
list (in-census for both models) and of --n-eval segments of its EVAL list (held out of both censuses)."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument('--names', required=True); ap.add_argument('--sup', required=True, type=Path)
ap.add_argument('--must', nargs='*', default=[]); ap.add_argument('--n-train', type=int, default=8)
ap.add_argument('--n-eval', type=int, default=4); ap.add_argument('--out', required=True)
ap.add_argument('--prefer-fruit', action='store_true', help='rank frames by fruit pixels first (fruit ids >= 10000)')
a = ap.parse_args()
d = json.load(open(a.names)); tr, ev = sorted(d['train']), sorted(d['eval'])
def score(n):
    f = a.sup / n
    if not f.exists(): return (False, 0, 0)
    m = np.array(Image.open(f), np.uint16); lab = m != 65535; ids = np.unique(m[lab])
    nt = int((ids < 10000).sum())
    if a.prefer_fruit: return (int(((m >= 10000) & lab).sum()) > 0, int(((m >= 10000) & lab).sum()), int(lab.sum()), nt)
    return (nt >= 2, int(lab.sum()), nt)
pick = {}
for split, lst, k in (('train', tr, a.n_train), ('eval', ev, a.n_eval)):
    for seg in np.array_split(np.array(lst), k):
        if len(seg) == 0: continue
        best = max(seg, key=score); s = score(best)
        if s[1] > 0: pick[str(best)] = {'split': split, 'labelled_px': s[-2], 'trees': s[-1]}
for n in a.must:
    s = score(n); pick[n] = {'split': 'train' if n in tr else ('eval' if n in ev else 'none'), 'labelled_px': s[-2], 'trees': s[-1], 'must': True}
frames = sorted(pick)
json.dump({'frames': frames, 'info': pick}, open(a.out, 'w'), indent=1)
print(f'[frames] {len(frames)}: ' + ' '.join(f"{n}({pick[n]['split']},{pick[n]['trees']}t)" for n in frames))
