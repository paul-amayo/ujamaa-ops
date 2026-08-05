#!/bin/bash
# Feature-field repair ladder vs the -1.6 TREE / -3.9 FRUIT regression
# (2026-08-05 (4)). Three runs, Paul's order: two-stage, warmup, weight.
#
# 1. stage2_frozen  — load the bg00 winner (patched ckpt with zero
#    high_features), freeze-geometry True: ONLY the 32-d features train.
#    Appearance provably untouched; 5k feature-only steps.
# 2. high_warmup6k  — fresh run, feature loss zeroed until step 6000
#    (HIGH_LOSS_WARMUP_STEP) so densification is appearance-driven.
# 3. high_w03       — fresh run, high-loss-weight 0.3 (was 1.0).
# All on the bg00 recipe (tree weights bg 0.0, sky 1.0, no depth), canary on.
# Gate: train TREE 19.14 / FRUIT 13.14 (stage2 must EQUAL it by construction;
# 2 and 3 must beat 17.59/9.22 and approach the gate).
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
cd /home/paperspace/code/nerf_new

COMMON="--pipeline.datamanager.semantic-dir $SUP \
  --pipeline.model.rasterize-mode antialiased \
  --pipeline.model.sky-loss-lambda 1.0 \
  --pipeline.model.report-masked-metrics True \
  --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10"

# ---- 1. two-stage: features on frozen bg00 geometry ----
T0=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_frozen \
    --load-dir $BD/stage2_init/nerfstudio_models \
    --pipeline.model.freeze-geometry True \
    --pipeline.model.high-loss-weight 1.0 \
    --max-num-iterations 20001 --steps-per-save 19998 \
    $COMMON || { echo "FEATFIX-FAIL stage2_frozen"; exit 1; }
echo "FEATFIX-DONE stage2_frozen in $(( ($(date +%s)-T0)/60 )) min"

# ---- 2. warmup: feature loss off until stop-split is over ----
T0=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
  TREE_WEIGHT_DIR=$SUP TREE_WEIGHT_BG=0.0 HIGH_LOSS_WARMUP_STEP=6000 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name high_warmup6k \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.stop-split-at 6000 \
    --max-num-iterations 15001 --steps-per-save 14998 \
    $COMMON || { echo "FEATFIX-FAIL high_warmup6k"; exit 1; }
echo "FEATFIX-DONE high_warmup6k in $(( ($(date +%s)-T0)/60 )) min"

# ---- 3. weight: high loss at 0.3 ----
T0=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
  TREE_WEIGHT_DIR=$SUP TREE_WEIGHT_BG=0.0 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name high_w03 \
    --pipeline.model.high-loss-weight 0.3 \
    --pipeline.model.stop-split-at 6000 \
    --max-num-iterations 15001 --steps-per-save 14998 \
    $COMMON || { echo "FEATFIX-FAIL high_w03"; exit 1; }
echo "FEATFIX-DONE high_w03 in $(( ($(date +%s)-T0)/60 )) min"
echo "=== FEATFIX-ALL-DONE $(date) ==="
