#!/usr/bin/env python3
"""Dump a side-car run's train / eval image names exactly as its datamanager splits them (dataparser only, no ckpt load).
nerf_new env:  pixi run python sidecar_split_names.py --config <run config.yml> --out <names.json>"""
import argparse, json, yaml
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
cfg = yaml.load(Path(a.config).read_text(), Loader=yaml.Loader)
dpc = cfg.pipeline.datamanager.dataparser
if getattr(cfg, 'data', None) is not None: dpc.data = Path(cfg.data)
dp = dpc.setup()
tr = [Path(f).name for f in dp.get_dataparser_outputs(split='train').image_filenames]
ev = [Path(f).name for f in dp.get_dataparser_outputs(split='test').image_filenames]
json.dump({'config': a.config, 'train': tr, 'eval': ev}, open(a.out, 'w'), indent=0)
print(f'[split] train {len(tr)} eval {len(ev)}; eval first {ev[:3]}; overlap {len(set(tr) & set(ev))}')
