"""Fitted TREE containment thresholds of a chunk side-car -> JSON {"tree_cuts": {tree_id: thr}} for the app (dashboard session,
2026-09-30 16:4x: tree queries lit a neighbouring canopy through the per-frame Otsu split; the app passes the fitted
threshold as the query's `cut` like the fruit cuts). Reads every TREE verdict line containment_eval wrote for this side-car
dir (tags "[<dir> sidecar bg_f1.0_r2 <frame>]", "[<dir> sidecar treecuts <frame>]", and the fruit-stage tags), keeps the
best-IoU threshold per tree id, leaves out nan or IoU < 0.05.
  python sidecar_tree_cuts.py <survey> <side-car dir name> <out json>"""
import json, re, sys
from pathlib import Path
sv, dn, out = sys.argv[1:4]
S = Path('/home/paperspace/data/citrus_all') / sv; O = S / 'experimental/h3dgs_sidecar_chunks' / dn
logs = [Path(f'/home/paperspace/logs/sidecar_{sv}_verdicts.log'), Path(f'/home/paperspace/logs/sidecar_{sv}_fruit_verdicts.log')]
pat = re.compile(r'^\[' + re.escape(dn) + r' sidecar (\S+) (kf_\d+\.png)\] TREE (\d+)\s+"([a-z]+)": thr ([0-9.]+|nan) IoU ([0-9.]+|nan)(?: prec ([0-9.]+|nan) rec ([0-9.]+|nan))?')
best = {}; seen = []
for lg in logs:
    if not lg.exists(): continue
    for ln in open(lg):
        m = pat.match(ln.strip())
        if not m: continue
        stage, frame, tid, word, thr, iou, prec, rec = m.groups(); tid = int(tid); seen.append((tid, frame))
        if thr == 'nan' or iou == 'nan' or float(iou) < 0.05: continue
        if tid not in best or float(iou) > best[tid]['iou']: best[tid] = {'thr': round(float(thr), 4), 'iou': round(float(iou), 4), 'prec': (round(float(prec), 3) if prec and prec != 'nan' else None), 'rec': (round(float(rec), 3) if rec and rec != 'nan' else None), 'word': word, 'frame': frame, 'stage': stage}
cuts = {str(k): v['thr'] for k, v in sorted(best.items())}
json.dump({'survey': sv, 'sidecar': str(O), 'tree_cuts': cuts, 'detail': {str(k): v for k, v in sorted(best.items())}, 'rule': 'best-IoU threshold per tree id over the scored frames; nan or IoU < 0.05 left out'}, open(out, 'w'), indent=1)
print(f'[tree-cuts] {dn}: {len(seen)} TREE verdict lines over {len({s[1] for s in seen})} frames -> {len(cuts)} tree cuts written to {out}: ' + ', '.join(f'{k} ({v["word"]}) thr {v["thr"]} IoU {v["iou"]}' for k, v in sorted(best.items())), flush=True)
left = sorted({s[0] for s in seen} - set(best))
if left: print(f'[tree-cuts]   left out (nan or IoU < 0.05): {left}', flush=True)
