#!/bin/bash
# REGRESSION CHECK: scene.json baseline (treelod_bg00_v1) + HiGH features ON.
# Single variable = the 32-d feature field (embedder + high loss + semantic
# dir). NO fruit-protect / fruit-weight here — those are the NEXT rung, so a
# regression is attributable to the feature field itself.
# Beat/hold: train TREE 19.14, train FRUIT 13.14 (treelod_bg00, 2026-08-05).
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 \
  HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth \
  TREE_WEIGHT_DIR=$SUP \
  TREE_WEIGHT_BG=0.0 \
  CANARY_EVERY=1000 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_BASELINE --experiment-name baseline_plus_high \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.datamanager.semantic-dir $SUP \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.stop-split-at 6000 \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --max-num-iterations 15001 --steps-per-save 14998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "BASE-HIGH-FAIL"; exit 1; }
echo "BASE-HIGH-DONE in $(( ($(date +%s)-T0)/60 )) min"
echo "=== BASELINE-PLUS-HIGH-ALL-DONE $(date) ==="
