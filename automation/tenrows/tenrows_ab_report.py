#!/usr/bin/env python3
"""A/B report for ten_rows block_013: keyframe arm vs full-frame arm.
Headline = ns-eval over all 13 held-out keyframes (stage1_eval.json per arm).
Curves = tensorboard: train PSNR (every logged step) and the single-image eval PSNR
(one random held-out image every 500 steps — noisy, shown as context only)."""
import json, glob, sys
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100")
ARMS = [("keyframes (87 train)", B / "block_013"), ("full frames (233 train)", B / "block_013_full")]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else B.parent.parent / "block_013_ab_report.png"

def scalars(run, tag):
    ea = EventAccumulator(str(run)); ea.Reload()
    if tag not in ea.Tags().get("scalars", []): return np.array([]), np.array([])
    ev = ea.Scalars(tag); return np.array([e.step for e in ev]), np.array([e.value for e in ev])

rows = []; fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)
for label, bd in ARMS:
    run = sorted(glob.glob(str(bd / "splat_runs_STAGE1/stage1/high/*/")))[-1]
    st, tr = scalars(run, "Train Metrics Dict/psnr"); se, ev = scalars(run, "Eval Images Metrics/psnr")
    ns = json.load(open(bd / "stage1_eval.json"))["results"] if (bd / "stage1_eval.json").exists() else {}
    tj = json.load(open(bd / "transforms.json"))
    rows.append((label, len(tj["train_filenames"]), len(tj["test_filenames"]),
                 np.mean(tr[st >= st.max() - 1000]) if len(tr) else np.nan,
                 np.mean(ev[-3:]) if len(ev) else np.nan, ns.get("psnr", np.nan), ns.get("ssim", np.nan), ns.get("lpips", np.nan)))
    ax[0].plot(st, tr, lw=1, label=label); ax[1].plot(se, ev, "o-", lw=1, ms=3, label=label)
ax[0].set_title("train PSNR (tensorboard)"); ax[1].set_title("single held-out image PSNR every 500 steps (noisy — context only)")
for a in ax: a.set_xlabel("step"); a.set_ylabel("PSNR (dB)"); a.grid(alpha=0.3); a.legend()
fig.suptitle("ten_rows block_013 — keyframes vs full-stream frames, identical 13 held-out keyframes, same LiDAR init, 15 001 iters")
fig.tight_layout(); fig.savefig(out)
print(f"{'arm':26s} {'train':>5s} {'test':>4s} {'trainPSNR':>9s} {'tb-eval':>8s} {'ns-eval PSNR':>12s} {'SSIM':>7s} {'LPIPS':>7s}")
for r in rows: print(f"{r[0]:26s} {r[1]:5d} {r[2]:4d} {r[3]:9.2f} {r[4]:8.2f} {r[5]:12.2f} {r[6]:7.4f} {r[7]:7.4f}")
if all(np.isfinite(r[5]) for r in rows): print(f"Δ ns-eval PSNR (full − kf) = {rows[1][5] - rows[0][5]:+.2f} dB")
print(f"[report] wrote {out}")
