"""Fitted fruit containment thresholds of a chunk side-car -> JSON {fruit_id: thr} for the app ("fruit is by containment",
Paul via the dashboard, 2026-09-30). Reads the fruit verdict lines that containment_eval wrote for this side-car dir
(tags "[<dir> sidecar fruitdensify <frame>]" and "[<dir> sidecar fruitcuts <frame>]"), maps the verdict's word back to the
fruit id through the side-car's supervision/trees_fruit_v3/manifest.json word_table (fruit ids are 10000 + fruit node id),
keeps the best-IoU threshold per fruit, and leaves out fruits whose verdict is nan or IoU < 0.05.
  python sidecar_fruit_cuts.py <survey> <side-car dir name> <out json>"""
import json, re, sys
from pathlib import Path
sv, dn, out = sys.argv[1:4]
S = Path('/home/paperspace/data/citrus_all') / sv; O = S / 'experimental/h3dgs_sidecar_chunks' / dn
VL = Path(f'/home/paperspace/logs/sidecar_{sv}_fruit_verdicts.log')
wt = json.load(open(O / 'supervision/trees_fruit_v3/manifest.json'))['word_table']; word2id = {v: int(k) for k, v in wt.items() if int(k) >= 10000}
pat = re.compile(r'^\[' + re.escape(dn) + r' sidecar (fruitdensify|fruitcuts) (kf_\d+\.png)\] FRUIT of (\d+) +"([a-z]+)": thr ([0-9.]+|nan) IoU ([0-9.]+|nan)')
best = {}; seen = []
for ln in open(VL):
    m = pat.match(ln.strip())
    if not m: continue
    stage, frame, tree, word, thr, iou = m.groups(); fid = word2id.get(word)
    seen.append((fid, tree, word, frame, thr, iou))
    if fid is None or thr == 'nan' or iou == 'nan' or float(iou) < 0.05: continue
    if fid not in best or float(iou) > best[fid]['iou']: best[fid] = {'thr': round(float(thr), 4), 'iou': round(float(iou), 4), 'tree': int(tree), 'word': word, 'frame': frame, 'stage': stage}
cuts = {str(k): v['thr'] for k, v in sorted(best.items())}
json.dump({'survey': sv, 'sidecar': str(O), 'fruit_cuts': cuts, 'detail': {str(k): v for k, v in sorted(best.items())}, 'source': str(VL), 'rule': 'best-IoU threshold per fruit id over the scored frames; nan or IoU < 0.05 left out'}, open(out, 'w'), indent=1)
print(f'[fruit-cuts] {dn}: {len(seen)} fruit verdict lines over {len({s[3] for s in seen})} frames -> {len(cuts)} fruit cuts written to {out}: ' + ', '.join(f'{k} (tree {v["tree"]} {v["word"]}) thr {v["thr"]} IoU {v["iou"]}' for k, v in sorted(best.items())), flush=True)
left = sorted({(s[0], s[1], s[2]) for s in seen if s[0] is not None and str(s[0]) not in cuts})
if left: print(f'[fruit-cuts]   left out (nan or IoU < 0.05): ' + ', '.join(f'{f} (tree {t} {w})' for f, t, w in left), flush=True)
