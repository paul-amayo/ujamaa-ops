"""Score the bilateral-grid splatfacto WITH its trained grid applied (nerfstudio applies it only in training mode, so ns-eval / ns-render
scored it raw): eval-mode render, then model._apply_bilateral_grid(rgb, cam_idx) exactly as get_outputs does in training; PSNR on the
same 60 training views as trainview_psnr.py, plus held-out raw. nerf_new pixi env: splat_bilateral_score.py <config.yml>"""
import math, re, sys
from pathlib import Path
import numpy as np, torch
from nerfstudio.utils.eval_utils import eval_setup
cfg = Path(sys.argv[1]); _, pipeline, _, _ = eval_setup(cfg, test_mode="test"); model = pipeline.model; dm = pipeline.datamanager; dev = model.device; tr, ev = dm.train_dataset, dm.eval_dataset
names = [Path(p).name for p in tr.image_filenames]; n = len(names); pick = sorted(range(n), key=lambda i: int(re.sub(r"\D", "", names[i]) or 0))[::max(1, n // 60)][:60]
psnr = lambda a, b: 10 * math.log10(1 / max(float(((a.clamp(0, 1) - b) ** 2).mean()), 1e-10)); model.eval(); raw, grid = [], []
with torch.no_grad():
    for i in pick:
        cam = tr.cameras[i:i + 1].to(dev); rgb = model.get_outputs(cam)["rgb"]; g = tr[i]["image"].to(dev)[..., :3]; H, W = rgb.shape[:2]
        raw.append(psnr(rgb, g)); grid.append(psnr(model._apply_bilateral_grid(rgb[None], i, H, W)[0], g))   # the grid slicer wants the batched [1,H,W,3] render get_outputs applies it to
    held = [psnr(model.get_outputs(ev.cameras[i:i + 1].to(dev))["rgb"], ev[i]["image"].to(dev)[..., :3]) for i in range(len(ev))]
print(f"[bilateral] {cfg.parent.parent.name}: training views (60): raw median {np.median(raw):.2f} -> WITH the trained bilateral grid {np.median(grid):.2f} (mean {np.mean(grid):.2f}); held-out raw median {np.median(held):.2f} mean {np.mean(held):.2f}   [H3DGS served: training 24.29 with exposure; splatfacto 30k + post-hoc affine 22.76; held-out H3DGS 15.76]", flush=True)
