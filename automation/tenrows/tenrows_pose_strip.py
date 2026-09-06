#!/usr/bin/env python3
"""One held-out keyframe of block_013 rendered by four stage-1 models (same split, 15k iters):
ours as-given | ours GL-flip only | roll-corrected (fix) | COLMAP poses. GT on the left.
Uses the last logged 'Eval Images/img' (GT | render) from each run's tensorboard."""
import glob, io, json, sys
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100"
arms = [("ours as-given (OpenCV c2w)", "block_013"), ("ours, GL flip only", "block_013_gl"), ("ours, roll-180 corrected", "block_013_fix"), ("COLMAP poses (reference)", "block_013_colmap")]
panels = []
for label, d in arms:
    runs = sorted(glob.glob(f"{B}/{d}/splat_runs_STAGE1/stage1/high/*/"))
    if not runs: continue
    ea = EventAccumulator(runs[-1], size_guidance={"images": 0}); ea.Reload(); ev = ea.Images("Eval Images/img")[-1]
    im = np.asarray(Image.open(io.BytesIO(ev.encoded_image_string)).convert("RGB")); W = im.shape[1] // 2
    m = json.load(open(f"{B}/{d}/stage1_eval.json"))["results"] if glob.glob(f"{B}/{d}/stage1_eval.json") else {}
    panels.append((f"{label}\nns-eval {m.get('psnr', float('nan')):.2f} dB  SSIM {m.get('ssim', float('nan')):.3f}  LPIPS {m.get('lpips', float('nan')):.3f}", im[:, W:], im[:, :W], ev.step))
fig, ax = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 3.2), dpi=130)
ax[0].imshow(panels[0][2]); ax[0].set_title("ground truth (held-out keyframe)", fontsize=9)
for a, (t, r, _, st) in zip(ax[1:], panels): a.imshow(r); a.set_title(t, fontsize=8)
for a in ax: a.set_axis_off()
fig.suptitle("ten_rows block_013 — same 13 held-out keyframes, same training split, 15 001 iters: what the pose frame does", fontsize=10)
fig.tight_layout(); out = sys.argv[1] if len(sys.argv) > 1 else "/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/block_013_pose_strip.png"; fig.savefig(out); print("[strip] wrote", out, "panels:", len(panels))
