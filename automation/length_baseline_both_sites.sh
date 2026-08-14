#!/bin/bash
# BLOCK-LENGTH BASELINE, BOTH SITES (2026-08-03, restarted clean).
# Fixes vs first attempt: every variant now has the parent's init_da3.ply
# (sub-blocks were silently falling back to RANDOM init -> invalid compare),
# masks stripped everywhere, and densification CAPPED at 6k so all variants
# work under one gaussian budget (also keeps runs ~20-25 min instead of 75).
#   citrus 04 b001: L124 31.9 m | L095 23.9 m | L073 18.3 m | L049 12.2 m
#   klapmuts b004 : L195 47.9 m | L100 25.1 m | L075 19.0 m | L050 12.5 m
# Splatfacto-equivalent path (no high features, no CLIP, no depth, no masks).
set -x
CRB=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F
KRB=/home/paperspace/data/klapmuts/apr_2026_zed/blocks_ns/lio_row
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
run () {  # blockdir name
  local BD=$1 NAME=$2
  local T0=$(date +%s)
  cd /home/paperspace/code/nerf_new
  echo "n" | MAX_JOBS=4 pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_BASE --experiment-name $NAME \
    --pipeline.model.enable-high-features False \
    --pipeline.model.high-loss-weight 0.0 \
    --pipeline.datamanager.semantic-dir $EMPTY \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.stop-split-at 6000 \
    --max-num-iterations 15001 --steps-per-save 14998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "BASE-FAIL $NAME"; return 1; }
  echo "BASE-DONE $NAME in $(( ($(date +%s)-T0)/60 )) min"
}
for v in L095 L124 L073 L049; do run $CRB/block_001_$v citrus_b001_$v; done
for v in L100 L195 L075 L050; do run $KRB/block_004_$v klap_b004_$v; done
echo "=== LENGTH-BASELINE-ALL-DONE $(date) ==="
