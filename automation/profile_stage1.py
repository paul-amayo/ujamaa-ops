#!/usr/bin/env python3
"""Offline per-frame FG-PSNR profile of a STAGE-1 run (no service, no census).

Usage: profile_stage1.py <block> <label> <run_dir_glob>
e.g.   profile_stage1.py 15 A0_random 'splat_runs_STAGE1/stage1_bg00/high/*'
Renders every keyframe with BlockRenderer on the run's own config and prints
the prod-bar row; writes lab_notebook/data/psnr_profiles/<label>_profiles.json.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer")
os.environ.setdefault("HIGH_EMBEDDER_CKPT",
    "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth")
from pathlib import Path
from PIL import Image
from block_renderer import BlockRenderer

b = int(sys.argv[1]); label = sys.argv[2]; glob = sys.argv[3]
R = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
GT = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3")
SKY = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/sky_masks")
OUT = Path("/home/paperspace/code/lab_notebook/data/psnr_profiles")
BD = R / f"block_{b:03d}"
run = sorted(BD.glob(glob))[-1]
print(f"[{label}] run {run}")
br = BlockRenderer(str(run / "config.yml"), block_id=str(b))
tj = json.loads((BD / "transforms.json").read_text())
frames = sorted(tj["frames"], key=lambda f: f["file_path"])
rows = []
for idx, fr in enumerate(frames):
    kf = fr["file_path"].split("/")[-1].split(".")[0]
    g, s = GT / f"{kf}.png", SKY / f"{kf}.png"
    if not (g.exists() and s.exists()):
        continue
    r = br.render_tensors(np.asarray(fr["transform_matrix"], float), 1280, 720, fovy=None)
    a = (r["rgb"].clamp(0, 1) * 255).cpu().numpy().astype(np.float64)
    gt = np.asarray(Image.open(g).convert("RGB").resize((1280, 720)), np.float64)
    mk = np.asarray(Image.open(s).convert("L").resize((1280, 720), Image.NEAREST)) < 127
    if mk.sum() < 1000:
        continue
    fg = 10 * np.log10(255**2 / max(((a - gt) ** 2)[mk].mean(), 1e-9))
    rows.append({"kf": kf, "idx": idx, "fg": round(fg, 2),
                 "holdout": idx % 10 == 0, "edge": idx < 2 or idx >= len(frames) - 2})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{label}_profiles.json").write_text(json.dumps({str(b): rows}))
tr = [r["fg"] for r in rows if not r["holdout"]]
inter = [r["fg"] for r in rows if not r["holdout"] and not r["edge"]]
med = float(np.median(tr))
print(f"[{label} b{b:03d}] med {med:.2f} P5 {np.percentile(tr,5):.2f} "
      f"int-min {min(inter):.2f} wvm {med-min(inter):.1f} "
      f"ho {np.median([r['fg'] for r in rows if r['holdout']]):.2f}")
worst = sorted([r for r in rows if not r["holdout"] and not r["edge"]], key=lambda r: r["fg"])[:3]
print(f"[{label}] worst interior: {[(r['kf'], r['fg']) for r in worst]}")
