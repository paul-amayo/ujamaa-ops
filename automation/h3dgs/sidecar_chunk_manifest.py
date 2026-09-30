"""Write a chunk side-car's supervision/trees_only/manifest.json: the union of the OWNING blocks' trees_only manifests
(word_table / level_table; scalar fields from the first block), the same owner rule as sidecar_chunk_dataset.py. render_service
resolves "tree N" -> word through this file (falling back from trees_fruit_v3/manifest.json), so without it every identity
query clears (dashboard session, 2026-09-30 16:1x, on chunk_0_0_expo). Words are survey-global per id (checked: one word per id
across all 05 blocks); a conflict is reported and the first block's word kept.
  python sidecar_chunk_manifest.py <survey> <O = side-car dir> [--cfg lio_row100] [--force]"""
import argparse, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('survey'); ap.add_argument('out'); ap.add_argument('--cfg', default='lio_row100'); ap.add_argument('--force', action='store_true'); a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; O = Path(a.out); D = O / 'supervision/trees_only'; dst = D / 'manifest.json'
names = {Path(f['file_path']).name for f in json.load(open(O / 'transforms.json'))['frames']}
man = None; wt = {}; lt = {}; conflicts = 0; blocks = []
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns' / a.cfg / 'block_[0-9][0-9][0-9]' / 'transforms.json'))):
    bd = Path(tj).parent; mf = bd / 'supervision/trees_only/manifest.json'
    if not mf.exists(): continue
    own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    if not (own & names): continue
    m = json.load(open(mf)); man = man or m; blocks.append(bd.name)
    for k, v in m.get('word_table', {}).items():
        if k in wt and wt[k] != v: conflicts += 1
        wt.setdefault(k, v)
    for k, v in m.get('level_table', {}).items(): lt.setdefault(k, v)
if dst.exists() and not a.force:
    old = json.load(open(dst)).get('word_table', {}); same = old == wt
    print(f'[chunk-manifest] {O.name}: manifest.json already present ({len(old)} ids) — {"identical word_table, left as is" if same else "DIFFERENT word_table (" + str(len(set(old.items()) ^ set(wt.items()))) + " differing entries), left as is; rerun with --force to replace"}', flush=True); raise SystemExit(0)
out = dict(man or {}); out.update({'word_table': wt, 'level_table': lt, 'assembled_from': blocks, 'assembled_by': 'sidecar_chunk_manifest.py', 'n_maps': len(list(D.glob('kf_*.png')))})
D.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(out, indent=1))
print(f'[chunk-manifest] {O.name}: trees_only/manifest.json written — word_table {len(wt)} ids from {len(blocks)} owning blocks, {conflicts} conflicts; keys {list(out.keys())[:8]}', flush=True)
