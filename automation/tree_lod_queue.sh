#!/bin/bash
# LOD-ON-TREES ladder (2026-08-04, Paul: "depth supervisions have been a
# bust... concentrate level of detail on trees only, we dont care about sky
# or ground"). NO depth supervision. Single variable = TREE_WEIGHT_BG:
# photometric L1 weight off-tree (tree/fruit id px always weigh 1.0), so
# gradient mass — and densification — concentrates on canopy.
#   treelod_bg030  bg 0.3   gentle re-prioritisation
#   treelod_bg010  bg 0.1   strong
#   treelod_bg000  bg 0.0   tree-only photometric (ground unconstrained)
# Sky supervision stays ON (it is how sky stays empty). Canary + step-0
# assert active (818d3fe). Score with score_splat.py: FG / TREE / FRUIT.
set -x
BD=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
cd /home/paperspace/code/nerf_new

for BG in 0.3 0.1 0.0; do
  TAG=treelod_bg$(echo $BG | tr -d '.')
  T0=$(date +%s)
  echo "n" | MAX_JOBS=4 \
    TREE_WEIGHT_DIR=$SUP \
    TREE_WEIGHT_BG=$BG \
    CANARY_EVERY=1000 \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_TREELOD --experiment-name $TAG \
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
      || { echo "TREELOD-FAIL $TAG"; exit 1; }
  echo "TREELOD-DONE $TAG in $(( ($(date +%s)-T0)/60 )) min"
done
echo "=== TREELOD-ALL-DONE $(date) ==="
