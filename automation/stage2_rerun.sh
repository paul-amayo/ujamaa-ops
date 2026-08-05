#!/bin/bash
# stage2_frozen RERUN after the trainer resume fix (nerf_new 156c09bf).
# v1 trained high_features against ORPHANED optimizer refs -> saved tensor
# was all-zero -> relevancy 0.0%. Same recipe, fixed loader.
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_frozen_v2 \
    --load-dir $BD/stage2_init/nerfstudio_models \
    --pipeline.model.freeze-geometry True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.datamanager.semantic-dir $SUP \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --max-num-iterations 20001 --steps-per-save 19998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "STAGE2V2-FAIL"; exit 1; }
echo "STAGE2V2-DONE in $(( ($(date +%s)-T0)/60 )) min"
echo "=== STAGE2V2-ALL-DONE $(date) ==="
