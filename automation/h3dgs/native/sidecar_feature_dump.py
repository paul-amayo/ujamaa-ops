#!/usr/bin/env python3
"""Dump a nerfstudio HiGH run's rendered features at named frames (UJAMAA, 2026-10-03) — exactly containment_eval.py's
render (render_frame at the 0.5 rescale), train OR eval split — into the npz layout hier_feature_render.py writes, so
both models are scored by the same `containment_eval.py --features-npz` code. One eval_setup for all frames.
nerf_new env:  HIGH_EMBEDDER_CKPT=<emb> pixi run python sidecar_feature_dump.py --config <run config.yml>
                  --frames kf_a.png ... --out-dir <dir> [--tag sidecar]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import render_frame
from nerfstudio.utils.eval_utils import eval_setup

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True); ap.add_argument('--frames', nargs='+', required=True)
ap.add_argument('--out-dir', required=True); ap.add_argument('--tag', default='sidecar')
a = ap.parse_args()
t0 = time.time()
_, pipe, _, step = eval_setup(Path(a.config))
model = pipe.model; model.eval(); model.step = step       # step=0 collapses SH (known gotcha)
dm = pipe.datamanager
sets = [('train', dm.train_dataset)] + ([('eval', dm.eval_dataset)] if getattr(dm, 'eval_dataset', None) is not None else [])
print(f'[dump] eval_setup {time.time() - t0:.0f}s; splits {[(s, len(d.image_filenames)) for s, d in sets]}', flush=True)
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
for name in a.frames:
    hit = None
    for split, ds in sets:
        nm = [Path(f).name for f in ds.image_filenames]
        if name in nm:
            hit = (split, ds, nm.index(name)); break
    if hit is None:
        print(f'[dump] {name}: in neither split — skipped', flush=True); continue
    split, ds, i = hit
    cam = ds.cameras[i:i + 1].to(model.device); cam.rescale_output_resolution(0.5)
    rgb, alpha, feat = render_frame(model, cam, model.config.lang_field_dim)
    out = Path(a.out_dir) / f'{name[:-4]}_{a.tag}.npz'
    np.savez_compressed(out, features=np.asarray(feat, np.float32), alpha=np.asarray(alpha, np.float32), rgb=rgb, split=split)
    fn = np.linalg.norm(feat, axis=-1)
    print(f'[dump] {name} ({split}): {feat.shape}, ||f||>0.5 on {(fn > 0.5).mean() * 100:.1f}% of px -> {out}', flush=True)
