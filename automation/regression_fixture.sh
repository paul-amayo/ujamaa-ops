#!/usr/bin/env bash
# Regression fixture: block_000 pass-0 depth-sup recipe (from run_pass0_depth_14k.sh).
# PASS = final TRAIN psnr_fg in [26.1, 27.5] (26.78 +/- 0.7, recalibrated 2026-07-13
# against the original 2026-06-23 pass0_depth_14k run on the identical recipe; the
# earlier 23.6-band figure was an EVAL-FG number from a different config).
# Run after any pipeline code change.
# Usage: regression_fixture.sh [--smoke]   (--smoke: 500 iters, no PSNR band check —
# verifies the chain runs, not the metric)
set -e
ROOT=/home/paperspace/data/citrus_all/03_13B_Jackal
DPH=$ROOT/blocks_ns/lio_arc_size15.0_ov0.10_kf20cm_dedup/block_000/splat_geometry_ablation/data_pass0_high
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
ITERS=14000; NAME=fixture_pass0
if [ "$1" = "--smoke" ]; then ITERS=500; NAME=fixture_smoke; fi
OUT=/home/paperspace/data/citrus_all/03_13B_Jackal/fixture_runs/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"
cd /home/paperspace/code/aru_sil_core
echo "n" | \
  DATAPARSER_SCALE=0.121442 \
  LIDAR_DEPTH_NPZ="$DPH/lidar_depth_x7m_morph.npz" \
  LIDAR_DEPTH_LAMBDA=0.5 \
  LIDAR_DEPTH_SMOOTH_LAMBDA=0 \
  MAX_JOBS=4 \
  pixi run --manifest-path "$NS_PIXI" ns-train high \
    --max-num-iterations $ITERS \
    --vis tensorboard \
    --output-dir "$OUT" \
    --experiment-name $NAME \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.background-color black \
    --pipeline.model.sky-loss-lambda 0.05 \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.report-masked-metrics True \
    nerfstudio-data --data "$DPH"
echo "FIXTURE RUN COMPLETE: $OUT"
echo "Now extract final psnr_fg from the tensorboard events under $OUT (psnr skill /"
echo "scripts/tb_fg_psnr.py) and check band [22.9, 24.3] unless --smoke."
