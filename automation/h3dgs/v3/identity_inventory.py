#!/usr/bin/env python3
"""identity_inventory.py — every v3 identity cell on disk with the inputs its scorer needs (paper E1, 2026-10-13).
Globs /home/paperspace/data for <ws>/identity/{text_bank.npz, features_B.bin.json, frames.json, maps_v3/}; the supervision dir and
the hierarchy come from the text bank's meta (manifest path, hierarchy_json), the embedder from features_B.bin.json — the ckpt the
seed was built against, never the newest one (embedder-era pairing). One JSON line per cell.
system python3:  identity_inventory.py [--out /home/paperspace/logs/identity_inventory.jsonl]"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/home/paperspace/logs/identity_inventory.jsonl'); a = ap.parse_args()
pats = ['/home/paperspace/data/*/*/experimental/v3/*/identity', '/home/paperspace/data/*/*/*/*/experimental/v3/*/identity']
rows = []
for d in sorted({p for pat in pats for p in glob.glob(pat)}):
    if not all(os.path.exists(f'{d}/{f}') for f in ('text_bank.npz', 'features_B.bin.json', 'frames.json', 'maps_v3')): continue
    meta = json.loads(str(np.load(f'{d}/text_bank.npz', allow_pickle=True)['meta'])); fb = json.load(open(f'{d}/features_B.bin.json'))
    parts = d.split('/'); survey = parts[parts.index('experimental') - 1]; ws = os.path.dirname(d)
    rows.append({'survey': survey, 'site': parts[4], 'ws': ws, 'chunk': os.path.basename(ws), 'identity': d, 'sup': os.path.dirname(meta['manifest']),
                 'hj': meta['hierarchy_json'], 'emb': fb['embedder'], 'gaussians': fb.get('n'), 'assigned': fb.get('assigned'),
                 'frames': len(json.load(open(f'{d}/frames.json'))['frames']), 'maps': len(glob.glob(f'{d}/maps_v3/*.npz')),
                 'missing': [k for k, p in (('sup', os.path.dirname(meta['manifest'])), ('hj', meta['hierarchy_json']), ('emb', fb['embedder'])) if not os.path.exists(p)]})
with open(a.out, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
by = {}
for r in rows: by.setdefault(r['survey'], []).append(r)
print(f'[inventory] {len(rows)} cells -> {a.out}'); [print(f'  {s:22s} {len(v):3d} cells, maps {sum(r["maps"] for r in v):4d}, missing inputs {sum(bool(r["missing"]) for r in v)}') for s, v in by.items()]
