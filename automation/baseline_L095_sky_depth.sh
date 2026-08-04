#!/bin/bash
# L095 + sky + DEPTH — single variable vs base_L095_sky.
# CRITICAL: uses lidar_depth_trainaligned.npz, NOT lidar_depth_morph.npz.
# high_model indexes the depth npz by batch["image_idx"] = the SPLIT-LOCAL
# dataset position, while the morph npz rows are in BLOCK frame order. With
# any eval split those disagree: on block_001 all 111 train frames were
# paired with a depth map from a median of 7 frames away (~1.75 m), max 13
# (~3.25 m). Every s6..s9b run and the klapmuts sweep trained on that.
# The aligned npz here has exactly the 85 train rows in train order.
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 \
  LIDAR_DEPTH_NPZ=$BD/lidar_depth_trainaligned.npz \
  LIDAR_DEPTH_LAMBDA=0.5 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_BASE --experiment-name base_L095_sky_depth \
    --pipeline.model.enable-high-features False \
    --pipeline.model.high-loss-weight 0.0 \
    --pipeline.datamanager.semantic-dir $EMPTY \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.stop-split-at 6000 \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --pipeline.model.output-depth-during-training True \
    --max-num-iterations 15001 --steps-per-save 14998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "DEPTH-FAIL"; exit 1; }
echo "BASE-DONE base_L095_sky_depth in $(( ($(date +%s)-T0)/60 )) min"
echo "=== L095-SKY-DEPTH-ALL-DONE $(date) ==="
