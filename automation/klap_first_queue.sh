#!/bin/bash
# KLAPMUTS-FIRST queue (Paul 2026-08-07): runs BEFORE the citrus night queue.
# Per row block 000-009: (A1) legacy colour PNGs if missing (sack identities,
# sam3_sack global ids), (A2) tree-only compiled id maps (colour_png_bridge),
# (A3) SAM3 BERRY ledger (image mode, in-tree prompting), (A4) final compile
# (sack trees + berry fruit) -> supervision/strict_tree_v2 with the v4_1k word
# table. Then (B) klapmuts embedder retrain on the new vocab (c20cos20 recipe,
# single-variable discipline), (C) stage1_bg00 for blocks without a checkpoint.
# Census-init stage2 for klapmuts is NOT queued yet (open Qs to Paul).
# Chains into night_stage_queue.sh when done.
set -x
GATE=/home/paperspace/logs/KLAP_FIRST
[ -f $GATE ] && mv $GATE $GATE.consumed
ROOT=/home/paperspace/data/klapmuts
ARU=/home/paperspace/code/aru_sil_core
SCR=$ARU/src/scripts
HJ=$ROOT/scene_graph/marker_hierarchy.json
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
SAM3_PIXI=/home/paperspace/code/sam3/pixi.toml
PYNS=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python

# ---- A. data: sack trees + berry fruit -> compiled supervision ----
for N in 000 001 002 003 004 005 006 007 008 009; do
  BD=$ROOT/blocks_ns/lio_row/block_$N
  [ -f "$BD/transforms.json" ] || { echo "KLAP-SKIP $N: no transforms"; continue; }
  cd $ARU
  [ "$(ls $BD/semantic_v2_B/*.png 2>/dev/null | wc -l)" -gt 10 ] || \
    $PYNS $SCR/save_semantic_pngs_fast.py --block-dir "$BD" --root "$ROOT" \
      --semantic-monolithic "$ROOT/filtered_semantic_v2.monolithic" \
      --global-ids "$ROOT/sam3_sack/global_ids.json" \
      || { echo "KLAP-FAIL $N: semantic pngs"; continue; }
  # tree-only id maps (fruit glob matches nothing yet)
  [ -f "$BD/supervision/trees_only/manifest.json" ] || \
    python3 $SCR/compile_supervision.py --block-dir "$BD" \
      --tree-source colour_png_bridge \
      --hierarchy $HJ \
      --fruit-ledger-glob "$ROOT/sam3_berry_b$N/clip_*/frame_entries.json" \
      --filter strict_fruit_tree_v1 \
      --out-dir "$BD/supervision/trees_only" \
      || { echo "KLAP-FAIL $N: tree compile"; continue; }
  # berry ledger: SAM3 image mode, prompted inside sack bboxes
  [ -f "$ROOT/sam3_berry_b$N/clip_000/frame_entries.json" ] || \
    pixi run --manifest-path $SAM3_PIXI python $SCR/fruit_in_trees_ledger.py \
      --data-dir "$ROOT" --block-dir "$BD" --out-name sam3_berry_b$N \
      --prompt berry --tree-idmap-dir "$BD/supervision/trees_only" \
      || echo "KLAP-WARN $N: berry ledger failed (tree-only supervision stands)"
  # final compile: sack trees + whatever berries survived the strict filter
  python3 $SCR/compile_supervision.py --block-dir "$BD" \
    --tree-source colour_png_bridge \
    --hierarchy $HJ \
    --fruit-ledger-glob "$ROOT/sam3_berry_b$N/clip_*/frame_entries.json" \
    --filter strict_fruit_tree_v1 \
    --out-dir "$BD/supervision/strict_tree_v2" \
    || echo "KLAP-FAIL $N: final compile"
done

# ---- B. klapmuts embedder retrain on v4_1k vocab (c20cos20 recipe) ----
cd /home/paperspace/code/nerf_new
if [ ! -f /home/paperspace/data/high/nerf/klapmuts_v2vocab1k/ckpts/model_best.pth ]; then
  pixi run python $ARU/src/interfaces/rerun/HiGH/train_hyperembedder_graph.py \
    --hierarchy-json $HJ --experiment-name klapmuts_v2vocab1k \
    --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 \
    --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
    --epochs 1500 --no-level-norms \
    > /home/paperspace/logs/embedder_klap_v2vocab1k.log 2>&1 \
    && echo "KLAP-DONE embedder" || echo "KLAP-FAIL embedder"
fi

# ---- C. stage1_bg00 for blocks without a checkpoint ----
for N in 000 001 002 003 004 005 006 007 008 009; do
  BD=$ROOT/blocks_ns/lio_row/block_$N
  [ -f "$BD/transforms.json" ] || continue
  if ls $BD/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
    echo "KLAP-SKIP $N: bg00 done"; continue
  fi
  TW=""
  [ -f "$BD/supervision/strict_tree_v2/manifest.json" ] && TW=$BD/supervision/strict_tree_v2
  T0=$(date +%s)
  echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
    TREE_WEIGHT_DIR=$TW TREE_WEIGHT_BG=0.0 \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_STAGE1 --experiment-name stage1_bg00 \
      --pipeline.model.enable-high-features False \
      --pipeline.model.high-loss-weight 0.0 \
      --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.stop-split-at 6000 \
      --pipeline.model.sky-loss-lambda 1.0 \
      --pipeline.model.report-masked-metrics True \
      --max-num-iterations 15001 --steps-per-save 14998 \
      --vis tensorboard nerfstudio-data \
      --eval-mode interval --eval-interval 10 \
    && echo "KLAP-DONE bg00 $N in $(( ($(date +%s)-T0)/60 )) min" \
    || echo "KLAP-FAIL bg00 $N"
done

echo "KLAP-QUEUE-COMPLETE — chaining citrus night queue"
exec /home/paperspace/code/automation/night_stage_queue.sh
