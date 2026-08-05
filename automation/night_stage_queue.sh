#!/bin/bash
# NIGHT QUEUE (Paul, 2026-08-05): first the FIXED two-stage rerun (v3), then
# STAGE 1 (scene-baseline appearance, treelod_bg00_v1 recipe) for every row
# block, surveys in Paul's order: 04, 05, 01, 02, 03.
#
# Inventory at queue time: 04 lio_row6F 6 | 05 lio_row 15 | 01 lio_row 35 |
# 02 none (redownload pending, skipped loudly) | 03 lio_row_halves 80.
# ~75 min/block: one night covers v3 + ~7 blocks; the queue is RESUMABLE —
# blocks with an existing stage1_bg00 run are skipped, so relaunching this
# script continues where the night ended.
#
# Recipe per block = scene.json baseline (bg00): tree-weighted L1 (bg 0.0)
# where the block has compiled supervision (TREE_WEIGHT_DIR set only then;
# loader disables itself loudly otherwise), sky loss 1.0 (no-op where no
# masks), stop-split 6000, 15k steps, NO depth, canary armed.
set -x
cd /home/paperspace/code/nerf_new
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
ROOT=/home/paperspace/data/citrus_all

# ---- 0. two-stage v3 with the freeze-after-load fix ----
BD=$ROOT/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP04=$ROOT/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
if [ ! -d "$BD/splat_runs_FEATFIX/stage2_frozen_v3" ]; then
  T0=$(date +%s)
  echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_frozen_v3 \
      --load-dir $BD/stage2_init/nerfstudio_models \
      --pipeline.model.freeze-geometry True \
      --pipeline.model.high-loss-weight 1.0 \
      --pipeline.datamanager.semantic-dir $SUP04 \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.sky-loss-lambda 1.0 \
      --pipeline.model.report-masked-metrics True \
      --max-num-iterations 20001 --steps-per-save 19998 \
      --vis tensorboard nerfstudio-data \
      --eval-mode interval --eval-interval 10 \
      && echo "NIGHT-DONE stage2_frozen_v3 in $(( ($(date +%s)-T0)/60 )) min" \
      || echo "NIGHT-FAIL stage2_frozen_v3"
fi

# ---- 1. stage-1 appearance for every row block, survey order 04,05,01,02,03 ----
for SPEC in "04_13D_Jackal:lio_row6F" "05_13D_Jackal:lio_row" \
            "01_13B_Jackal:lio_row" "02_13B_Jackal:lio_row" \
            "03_13B_Jackal:lio_row_halves"; do
  S=${SPEC%%:*}; SCHEME=${SPEC##*:}
  if ! ls -d $ROOT/$S/blocks_ns/$SCHEME/block_[0-9][0-9][0-9] >/dev/null 2>&1; then
    echo "NIGHT-SKIP $S: no $SCHEME blocks (02 = redownload pending)"
    continue
  fi
  for B in $ROOT/$S/blocks_ns/$SCHEME/block_[0-9][0-9][0-9]; do
    N=$(basename $B)
    [ -f "$B/transforms.json" ] || { echo "NIGHT-SKIP $S/$N: no transforms"; continue; }
    [ -d "$B/splat_runs_STAGE1/stage1_bg00" ] && { echo "NIGHT-SKIP $S/$N: done"; continue; }
    TW=""
    [ -f "$B/supervision/strict_tree_v2/manifest.json" ] && TW=$B/supervision/strict_tree_v2
    T0=$(date +%s)
    echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
      TREE_WEIGHT_DIR=$TW TREE_WEIGHT_BG=0.0 \
      pixi run ns-train high \
        --data $B --output-dir $B/splat_runs_STAGE1 --experiment-name stage1_bg00 \
        --pipeline.model.enable-high-features False \
        --pipeline.model.high-loss-weight 0.0 \
        --pipeline.datamanager.semantic-dir $EMPTY \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.stop-split-at 6000 \
        --pipeline.model.sky-loss-lambda 1.0 \
        --pipeline.model.report-masked-metrics True \
        --max-num-iterations 15001 --steps-per-save 14998 \
        --vis tensorboard nerfstudio-data \
        --eval-mode interval --eval-interval 10 \
        && echo "NIGHT-DONE $S/$N in $(( ($(date +%s)-T0)/60 )) min" \
        || echo "NIGHT-FAIL $S/$N"
  done
done
echo "=== NIGHT-QUEUE-ALL-DONE $(date) ==="
