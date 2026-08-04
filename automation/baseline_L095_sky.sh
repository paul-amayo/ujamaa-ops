#!/bin/bash
# L095 + SKY SUPERVISION — single-variable vs the mask-free L095 baseline.
# Identical in every other respect (splatfacto-equivalent path, LiDAR init,
# stop-split 6000, 15k, eval-interval 10). Adds ONLY:
#   mask_path on all 95 frames (SAM3 sky -> fg masks)  => sky excluded from RGB loss
#   --pipeline.model.sky-loss-lambda 1.0               => alpha penalised at sky px
#   --pipeline.model.report-masked-metrics True        => psnr_fg logged
# Baseline to beat (mask-free L095): train 21.03 / eval 10.47 dB.
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 pixi run ns-train high \
  --data $BD --output-dir $BD/splat_runs_BASE --experiment-name base_L095_sky \
  --pipeline.model.enable-high-features False \
  --pipeline.model.high-loss-weight 0.0 \
  --pipeline.datamanager.semantic-dir $EMPTY \
  --pipeline.model.rasterize-mode antialiased \
  --pipeline.model.stop-split-at 6000 \
  --pipeline.model.sky-loss-lambda 1.0 \
  --pipeline.model.report-masked-metrics True \
  --max-num-iterations 15001 --steps-per-save 14998 \
  --vis tensorboard nerfstudio-data \
  --eval-mode interval --eval-interval 10 || { echo "SKY-FAIL"; exit 1; }
echo "BASE-DONE base_L095_sky in $(( ($(date +%s)-T0)/60 )) min"
echo "=== L095-SKY-ALL-DONE $(date) ==="
