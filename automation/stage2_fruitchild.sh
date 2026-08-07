#!/bin/bash
# REAL fruit-as-children training (2026-08-06). The no-walk test proved the
# field learns fruit DIRECTION but not DEPTH: rendered features at fruit px
# sit at tree radius (no-walk recall 0%), and only the walk's ray-extension
# faked 88.6% while manufacturing 54% canopy FP (100% of FPs = own tree's
# fruit — walk degeneracy, not field contamination).
# This run: stage2 on frozen bg00 geometry, targets rebuilt through the
# per-word node_types path (cache key node_types_v2 forces it), and the
# fruit minority (350 px vs 82k) weighted so it can pull rendered features
# to fruit depth: high-loss-fruit-weight 10 (norm-gated: targets past 6.0
# tangent norm = fruit) + fruit-protect feature anchoring.
# Verdict tooling: fruit_pointing_map.py in BOTH walk and --no-walk modes;
# the number that must move is NO-WALK recall (0% -> up) at no-walk FP ~0.
set -x
BD=${STAGE2_BD:-/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky}
SUP=${STAGE2_SUP:-/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2}
EMB=${STAGE2_EMBEDDER:-/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth}
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
  pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name ${STAGE2_NAME:-stage2_fruitchild} \
    --load-dir ${STAGE2_INIT_DIR:-$BD/stage2_init/nerfstudio_models} \
    --pipeline.model.freeze-geometry True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight ${STAGE2_FRUIT_W:-10.0} \
    --pipeline.model.fruit-protect True \
    --pipeline.datamanager.semantic-dir $SUP \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --max-num-iterations 20001 --steps-per-save 19998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 \
    && echo "FRUITCHILD-DONE in $(( ($(date +%s)-T0)/60 )) min" \
    || echo "FRUITCHILD-FAIL"
# resume the night queue behind it (skip-checks continue at 05)
exec /home/paperspace/code/automation/night_stage_queue.sh
