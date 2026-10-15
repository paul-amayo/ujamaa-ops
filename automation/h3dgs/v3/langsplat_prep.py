#!/usr/bin/env python3
"""langsplat_prep.py — a LangSplat dataset from a v3 chunk workspace (paper E3, 2026-10-13). LangSplat's COLMAP reader trains on every
image registered in <dataset>/sparse/0 (its only split is every-8th), so the baseline gets its own dataset dir with ONLY our training
split (split.json 'train', optionally every N-th of them, plus the census frames the verdicts use) as symlinked images and a filtered
images.bin; cameras.bin and points3D.bin are the workspace's. Held-out frames never enter it, exactly as in the v3 training.
system python3:  langsplat_prep.py --ws <chunk ws> [--every 2] [--keep <frames.json of the identity cell>] [--out <ws>/langsplat]"""
import argparse, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, write_images_binary
ap = argparse.ArgumentParser(); ap.add_argument('--ws', required=True, type=Path); ap.add_argument('--every', type=int, default=1); ap.add_argument('--keep', default='', help='frames.json: its training-split frames are always kept')
ap.add_argument('--out', default=None, type=Path); a = ap.parse_args(); out = a.out if a.out is not None else a.ws / 'langsplat'   # Path('') is '.', never a default
split = json.load(open(a.ws / 'split.json')); train = list(split['train']); held = set(split.get('test', split.get('eval', []))); keep = set()   # 05 1_0's older ws names the held-out key differently
if a.keep:
    fj = json.load(open(a.keep)); keep = {f for f in fj['frames'] if f in train and f not in held}   # census frames of the cell that are training views
chosen = {f for i, f in enumerate(train) if i % a.every == 0} | (keep & set(train))
imgs = read_images_binary(str(a.ws / 'sparse/0/images.bin'))
sub = {k: v for k, v in imgs.items() if os.path.basename(v.name) in chosen}
missing = chosen - {os.path.basename(v.name) for v in sub.values()}
(out / 'images').mkdir(parents=True, exist_ok=True); (out / 'sparse/0').mkdir(parents=True, exist_ok=True)
for v in sub.values():
    dst = out / 'images' / os.path.basename(v.name)
    if not dst.exists(): os.symlink((a.ws / 'images' / os.path.basename(v.name)).resolve(), dst)
write_images_binary(sub, str(out / 'sparse/0/images.bin'))
for f in ('cameras.bin', 'points3D.bin'):
    dst = out / 'sparse/0' / f
    if not dst.exists(): shutil.copy(a.ws / 'sparse/0' / f, dst)
json.dump({'ws': str(a.ws), 'every': a.every, 'train_total': len(train), 'chosen': sorted(chosen), 'kept_census': sorted(keep & set(train)), 'missing_in_sparse': sorted(missing)}, open(out / 'subset.json', 'w'), indent=1)
print(f'[langsplat-prep] {a.ws.name}: {len(sub)} of {len(train)} training images (every {a.every} + {len(keep & set(train))} census frames; {len(missing)} missing in sparse), {len(imgs)} registered in the ws -> {out}', flush=True)
