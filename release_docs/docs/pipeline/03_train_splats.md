# 3 · Training the reconstruction

**Environment: pixi (`pixi run` / `pixi shell`).** One block at a time;
~6 minutes per block on an A100-40 (15k iterations), ~16 GB VRAM.

Each block trains in **two stages**:

1. **Stage 1 — geometry**: gaussian-splat training from the block's
   keyframes, LiDAR-initialised, sky loss on, no semantic features.
2. **Stage 2 — identity features (census-init)**: the stage-1 geometry is
   frozen and per-gaussian identity features are trained in, initialised
   by a *census*: every gaussian is assigned to the plant whose supervision
   it actually interacted with during rasterisation, by majority. This is
   what lets the twin answer "which tree is this?" — the viewer and panels
   query those features.

```bash
pipeline/train/train_block.sh data/<survey_id> <block_number>
```
<!-- CONSOLIDATION-CONTRACT: today prod_block_recipe.sh (init → stage1 →
censusinit_block.sh → chain → QA gate), paths generalised. -->

The driver runs init → stage 1 → census/stage 2 → a quality gate, per
block; `--all` queues the fleet.

## The QA gate (built in — read its output, don't skip it)

Numbers, not visual inspection. Per block, from held-out views:

- held-out PSNR: median ≥ 21, 5th-percentile ≥ 17, interval-min ≥ 16
  (thresholds are the shipped defaults; they came from a 43-block
  production fleet)
- identity check: the block's plants point at the right census entries

**On a failed gate the driver re-rolls stage 1 once.** Training variance
is real — we measured 2–9 dB swings on a few frames across identical
re-runs — and one re-roll is the measured fix. What we tested and
*rejected*, so you don't have to: geometric culls (−2.6 dB collateral),
view-exclusivity culls (no effect), the in-loop camera optimiser
(−3.7 dB). A block that fails twice is flagged and left for triage — the
usual cause is upstream (poses, a glare pass), not training.

## Verify

```bash
pixi run python pipeline/train/report_fleet.py data/<survey_id>
```

Prints the per-block table (PSNR/SSIM/LPIPS, wall time) and writes the
fleet report figure. Our reference numbers on a 23-block survey with
refined poses: **median 18.6 dB held-out, min 16.3, max 21.3**. If your
median is under ~16 with refined poses, suspect capture (glare, speed) or
the §2.1 frame check — not the trainer.

## Common failures

- *Out of VRAM*: another process holds the GPU (the serving stack
  preloads blocks — stop it while training, or cap its budget).
- *A block 3+ dB under its neighbours*: look at its keyframes first;
  one backlit pass explains most outliers.
- *Stage 2 census looks scrambled*: the supervision inputs and the
  hierarchy (next stage) are from different runs — the census must be
  built against the same detection vocabulary it will be queried with.

Next: [04_hierarchy.md](04_hierarchy.md).
