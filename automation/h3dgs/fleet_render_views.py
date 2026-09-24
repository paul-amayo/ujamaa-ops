#!/usr/bin/env python
"""Render a per-block nerfstudio (HiGH/splatfacto) checkpoint at named keyframes, saving <kf name>.png, so the
per-block fleet can be put side by side with the hierarchy on the SAME held-out views.
  python fleet_render_views.py <config.yml> <out_dir> <kf name> [<kf name> ...]      (run with nerf_new's pixi python)
"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
cfg, out = Path(sys.argv[1]), Path(sys.argv[2]); wanted = set(sys.argv[3:]); out.mkdir(parents=True, exist_ok=True)
config, pipeline, ckpt, step = eval_setup(cfg, test_mode="inference")
print(f"[fleet-render] {cfg.parent.name} step {step}", flush=True)
done = 0
for split, ds in [("eval", pipeline.datamanager.eval_dataset), ("train", pipeline.datamanager.train_dataset)]:
    names = [Path(str(p)).name for p in ds.image_filenames]
    for i, n in enumerate(names):
        if n not in wanted or (out / n).exists(): continue
        cam = ds.cameras[i : i + 1].to(pipeline.device)
        with torch.no_grad(): o = pipeline.model.get_outputs_for_camera(cam)
        rgb = (o["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8); Image.fromarray(rgb).save(out / n)
        gt = ds.get_image_float32(i) if hasattr(ds, "get_image_float32") else None
        if gt is not None:
            g = gt.cpu().numpy() if torch.is_tensor(gt) else gt; g = g[..., :3]; mse = float(np.mean((g - rgb / 255.0) ** 2)); print(f"[fleet-render] {n} ({split}) PSNR {10*np.log10(1/max(mse,1e-12)):.2f} dB", flush=True)
        done += 1
print(f"[fleet-render] rendered {done}/{len(wanted)} requested views -> {out}", flush=True)
