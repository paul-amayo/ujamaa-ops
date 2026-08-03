#!/bin/bash
# SPLATFACTO-EQUIVALENT BASELINE PATH for HiGH (2026-08-03, Paul's request).
# Purpose: fast, honest reconstruction baselines. Strips every HiGH extra:
#   enable-high-features False -> no 32-D param, no extra optimizer group,
#     no densification bookkeeping for it, NO feature rasterization pass
#   high-loss-weight 0         -> no geodesic loss
#   semantic-dir = empty dir   -> datamanager goes RGB-only, NO CLIP cache
#   no LIDAR_DEPTH_NPZ         -> no depth supervision
#   sky-loss-lambda 0 (default), no fg masks (stripped from transforms)
#   fruit-protect / fruit-densify off (defaults)
# Usage: baseline_rgb.sh <block-dir> <run-name> [iters]
set -x
BD=$1; NAME=$2; ITERS=${3:-15000}
EMPTY=/home/paperspace/logs/empty_semantic
mkdir -p $EMPTY
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 pixi run ns-train high \
  --data $BD --output-dir $BD/splat_runs_BASE --experiment-name $NAME \
  --pipeline.model.enable-high-features False \
  --pipeline.model.high-loss-weight 0.0 \
  --pipeline.datamanager.semantic-dir $EMPTY \
  --pipeline.model.rasterize-mode antialiased \
  --max-num-iterations $((ITERS+1)) --steps-per-save $((ITERS-2)) \
  --vis tensorboard nerfstudio-data \
  --eval-mode interval --eval-interval 10 || { echo "BASE-FAIL $NAME"; exit 1; }
T1=$(date +%s)
echo "BASE-TRAINED $NAME in $(( (T1-T0)/60 )) min for $ITERS iters"
echo "BASE-ALL-DONE $NAME"
