#!/usr/bin/env python3
"""Score a trained splat the ONLY way that is meaningful once masks exist.

RULE (Paul, 2026-08-04): when sky/fg masks are in play, FULL-IMAGE PSNR IS
MEANINGLESS and must not be quoted. A masked run deliberately leaves sky
unfitted, so full-image PSNR punishes it for doing the right thing — it
misled us twice (2026-08-03 (10)/(11), and again on the sky-vs-no-sky
comparison). This tool reports:

  FG    PSNR over fg-mask pixels          (the reconstruction number)
  FRUIT PSNR over fruit-supervised pixels (the demo-differentiator number)

both split into train / eval, over a FIXED frame set so runs are comparable.
Fruit pixels come from the compiled supervision id maps (ids >= 200), looked
up BY FRAME NAME so sub-block variants score against the same ground truth.

  usage: score_splat.py <run_dir> [<run_dir> ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup

torch.set_grad_enabled(False)

D = Path("/home/paperspace/data/citrus_all/04_13D_Jackal")
KF = D / "kf_images"
FG = D / "fg_masks"
SUP = D / "blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2"
UNLAB = 65535


def masks_for(name, W, H):
    fgm = None
    p = FG / name
    if p.exists():
        fgm = np.array(Image.open(p).resize((W, H), Image.NEAREST)) > 127
    frm = None
    q = SUP / name
    if q.exists():
        a = np.array(Image.open(q), np.uint16)
        if a.shape != (H, W):
            a = np.array(Image.fromarray(a).resize((W, H), Image.NEAREST))
        frm = (a >= 200) & (a != UNLAB)
        if not frm.any():
            frm = None
    return fgm, frm


def psnr(se, n):
    return -10 * np.log10(max(se / max(n, 1), 1e-10)) if n else float("nan")


def score(run_dir):
    cfg = sorted(glob.glob(f"{run_dir}/high/*/config.yml"))[-1]
    _, pipe, _, step = eval_setup(Path(cfg))
    m = pipe.model
    m.eval()
    out = {}
    for split in ("train", "eval"):
        ds = getattr(pipe.datamanager, f"{split}_dataset", None)
        if ds is None:
            continue
        names = [Path(f).name for f in ds.image_filenames]
        sf = nf = sr = nr = 0.0
        n_fruit_frames = 0
        for i, nm in enumerate(names):
            o = m.get_outputs_for_camera(ds.cameras[i:i + 1].to(m.device))
            rgb = o["rgb"].clamp(0, 1).cpu().numpy()
            H, W, _ = rgb.shape
            gt = np.asarray(Image.open(KF / nm).convert("RGB")
                            .resize((W, H))).astype(np.float32) / 255
            e = (rgb - gt) ** 2
            fgm, frm = masks_for(nm, W, H)
            if fgm is None:
                fgm = np.ones((H, W), bool)
            sf += e[fgm].sum(); nf += fgm.sum() * 3
            if frm is not None:
                sr += e[frm].sum(); nr += frm.sum() * 3
                n_fruit_frames += 1
        out[split] = (psnr(sf, nf), psnr(sr, nr), len(names), n_fruit_frames,
                      int(nr / 3))
    del pipe, m
    torch.cuda.empty_cache()
    return step, out


if __name__ == "__main__":
    print(f"{'run':34s}{'split':6s}{'FG dB':>8s}{'FRUIT dB':>10s}"
          f"{'frames':>8s}{'fruit-fr':>9s}{'fruit-px':>10s}")
    for rd in sys.argv[1:]:
        step, res = score(rd)
        tag = Path(rd).name[:33]
        for split, (fg, fr, n, nff, npx) in res.items():
            print(f"{tag:34s}{split:6s}{fg:8.2f}{fr:10.2f}{n:8d}{nff:9d}{npx:10d}")
            tag = ""
