#!/usr/bin/env python3
"""Recipe v3 identity cell from EXISTING per-keyframe supervision maps (2026-10-12; generalises apr_lane_native_prep.py 'cell').
A v3 workspace (build_chunk_ws.py: split.json train/test frame names) + supervision/trees_only dirs that already hold id maps for the
survey's keyframes (e.g. the blocks' maps the block process compiled) -> <cell>/supervision/trees_only (links named by WORKSPACE frame,
merged manifest: word_table / level_table union, a clash aborts), split_names.json (census = workspace training frames with a map,
eval = workspace held-out frames with a map). Lanes train on the full 15 Hz stream (f_XXXXX.png); their keyframes are frames of that
stream, matched by exact stamp in a kf_map {kf_XXXXXX.png: f_XXXXX.png} (apr_lane_native_prep.py kfmap) -> --kf-map.
System python3:  v3_cell_from_maps.py --ws <workspace> --sup-glob '<glob of trees_only dirs>' --out <cell> [--kf-map kf_map.json]"""
import argparse, glob, json, os
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--ws', required=True); ap.add_argument('--sup-glob', required=True); ap.add_argument('--out', required=True); ap.add_argument('--kf-map', default='')
a = ap.parse_args(); WS, OUT = Path(a.ws), Path(a.out); SUP = OUT / 'supervision/trees_only'; SUP.mkdir(parents=True, exist_ok=True)
split = json.load(open(WS / 'split.json')); km = json.load(open(a.kf_map)) if a.kf_map else None
frame_to_map_name = {v: k for k, v in km.items()} if km else None
src, wt, lt, man0, dirs = {}, {}, {}, None, sorted(glob.glob(a.sup_glob))
for d in dirs:
    d = Path(d)
    if not (d / 'manifest.json').exists(): continue
    m = json.load(open(d / 'manifest.json')); man0 = man0 or m
    for k, v in m.get('word_table', {}).items():
        assert wt.get(k, v) == v, f'word clash {k}: {wt[k]} vs {v} ({d})'; wt[k] = v
    lt.update(m.get('level_table', {}))
    for p in d.glob('*.png'): src.setdefault(p.name, p)
assert man0 is not None, f'no manifest under {a.sup_glob}'
def map_for(frame):
    name = frame_to_map_name.get(frame) if frame_to_map_name is not None else frame
    return src.get(name) if name else None
train, ev = [], []
for lst, names in ((train, split['train']), (ev, split['test'])):
    for f in names:
        p = map_for(f)
        if p is None: continue
        dst = SUP / f
        if not dst.exists(): os.symlink(p, dst)
        lst.append(f)
man = dict(man0); man['word_table'] = wt; man['level_table'] = lt
man['note'] = f'merged from {len(dirs)} supervision dirs for v3 workspace {WS} (v3_cell_from_maps.py){"; keyframes matched to stream frames by kf_map" if km else ""}'
json.dump(man, open(SUP / 'manifest.json', 'w'), indent=1)
json.dump({'source': f'{WS}/split.json x maps of {a.sup_glob}', 'train': train, 'eval': ev}, open(OUT / 'split_names.json', 'w'), indent=0)
print(f'[v3-cell] {OUT}: {len(train)} census + {len(ev)} held-out frames with maps (workspace {len(split["train"])} / {len(split["test"])}); '
      f'manifest {len(wt)} words from {len(dirs)} dirs; {len(src)} maps available')
