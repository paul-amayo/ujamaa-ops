"""Assemble a chunk side-car's FRUIT supervision (supervision/trees_fruit_v3) from the blocks that OWN each of its keyframes,
the same owner rule as sidecar_chunk_dataset.py (a keyframe's id map comes from the block whose transforms.json lists it;
context-only maps only as a fallback), plus a merged manifest.json (word_table / level_table = the union over the owning
blocks; fruit ids are survey-global, 10000 + fruit index, and the tables agreed on every id when checked on 05).
  python sidecar_chunk_fruitsup.py <survey> <O = side-car dir> [--cfg lio_row100]"""
import argparse, glob, json, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('survey'); ap.add_argument('out'); ap.add_argument('--cfg', default='lio_row100'); a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; O = Path(a.out); D = O / 'supervision/trees_fruit_v3'; D.mkdir(parents=True, exist_ok=True)
names = {Path(f['file_path']).name for f in json.load(open(O / 'transforms.json'))['frames']}
sup = {}; extra = {}; man = None; wt = {}; lt = {}; conflicts = 0; blocks = set()
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns' / a.cfg / 'block_[0-9][0-9][0-9]' / 'transforms.json'))):
    bd = Path(tj).parent; fd = bd / 'supervision/trees_fruit_v3'
    if not (fd / 'manifest.json').exists(): continue
    own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    m = json.load(open(fd / 'manifest.json')); man = man or m
    used = False
    for f in sorted(fd.glob('kf_*.png')):
        if f.name not in names: continue
        if f.name in own: sup[f.name] = f; used = True
        else: extra.setdefault(f.name, f)
    if used:
        blocks.add(bd.name)
        for k, v in m.get('word_table', {}).items():
            if k in wt and wt[k] != v: conflicts += 1
            wt.setdefault(k, v)
        for k, v in m.get('level_table', {}).items(): lt.setdefault(k, v)
for n, f in extra.items(): sup.setdefault(n, f)
n_new = 0
for n, f in sup.items():
    dst = D / n
    if not dst.exists(): os.link(f, dst); n_new += 1
out = dict(man or {}); out.update({'word_table': wt, 'level_table': lt, 'assembled_from': sorted(blocks), 'assembled_by': 'sidecar_chunk_fruitsup.py', 'n_maps': len(sup)})
(D / 'manifest.json').write_text(json.dumps(out, indent=1))
fruit_ids = sorted(int(k) for k in wt if int(k) >= int(out.get('fruit_id_base', 10000)))
print(f'[chunk-fruitsup] {O.name}: {len(sup)} of {len(names)} keyframes have a fruit map ({n_new} linked now; {len(extra)} context-only fallbacks) from {len(blocks)} owning blocks; word_table {len(wt)} ids ({len(fruit_ids)} fruit ids), {conflicts} conflicts -> {D}', flush=True)
