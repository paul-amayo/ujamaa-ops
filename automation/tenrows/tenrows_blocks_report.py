#!/usr/bin/env python3
"""Per-block stage-1 summary for ten_rows: ns-eval PSNR (all held-out views) per block,
keyframe counts, wall time. Inputs: ~/logs/tenrows_stage1_psnr.tsv + <block>/stage1_eval.json."""
import json, sys, csv
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else B.parent.parent / "stage1_blocks_report.png"
tsv = {r["block"]: r for r in csv.DictReader(open("/home/paperspace/logs/tenrows_stage1_psnr.tsv"), delimiter="\t")}
rows = []
for bd in sorted(B.glob("block_0??")):
    n = bd.name; r = tsv.get(n, {}); ev = bd / "stage1_eval.json"
    m = json.load(open(ev))["results"] if ev.exists() else {}
    ntr = len(json.load(open(bd / "transforms.json"))["frames"])
    rows.append((n, ntr, float(r.get("wall_s", "nan")), float(r.get("train_psnr_last1k", "nan")), m.get("psnr", np.nan), m.get("ssim", np.nan), m.get("lpips", np.nan)))
print(f"{'block':10s} {'kf':>4s} {'wall_s':>7s} {'trainPSNR':>9s} {'evalPSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s}")
for r in rows: print(f"{r[0]:10s} {r[1]:4d} {r[2]:7.0f} {r[3]:9.2f} {r[4]:8.2f} {r[5]:7.4f} {r[6]:7.4f}")
ev = np.array([r[4] for r in rows]); wall = np.array([r[2] for r in rows]); kf = np.array([r[1] for r in rows])
ok = np.isfinite(ev)
print(f"[blocks] n={ok.sum()} eval PSNR median {np.nanmedian(ev):.2f} (min {np.nanmin(ev):.2f}, max {np.nanmax(ev):.2f}); wall median {np.nanmedian(wall):.0f} s, total {np.nansum(wall)/3600:.2f} h; kf total {kf.sum()}")
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), dpi=120)
x = np.arange(len(rows)); ax[0].bar(x, ev, color="#4a8"); ax[0].set_xticks(x); ax[0].set_xticklabels([r[0][-3:] for r in rows], rotation=90)
for i, r in enumerate(rows): ax[0].text(i, (ev[i] if np.isfinite(ev[i]) else 0) + 0.2, f"{r[1]}", ha="center", fontsize=7)
ax[0].set_ylabel("ns-eval PSNR (dB), all held-out views"); ax[0].set_title("stage-1 eval PSNR per block (label = keyframes)"); ax[0].grid(axis="y", alpha=0.3)
ax[1].scatter(kf, wall / 60, c="#c60"); ax[1].set_xlabel("keyframes in block"); ax[1].set_ylabel("wall time (min)"); ax[1].set_title("stage-1 wall time vs block size (15 001 iters)"); ax[1].grid(alpha=0.3)
fig.suptitle("dec_2025_ten_rows — stage-1 (no supervision, sky loss off, LiDAR init, corrected poses)"); fig.tight_layout(); fig.savefig(out); print(f"[report] wrote {out}")
