#!/bin/bash
# FRUIT-IN-TREES repair pass (2026-08-02) — gated on CHAIN-04-ALL-DONE.
# chain_04's fruit stage prompted FULL frames (regression vs the canonical
# order: trees first, fruit WITHIN trees, upscale 2.0 + parent rejection).
# This pass redoes fruit per block with fruit_in_trees_ledger.py (tree crops
# from the projected id maps), recompiles, and retrains s7_bNNN ONLY where
# strict keeps exceed the proven-dead floor (>25; s5/s6 collapsed at <=25).
set -x
ARU=/home/paperspace/code/aru_sil_core
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
SAM3_PIXI=/home/paperspace/code/sam3/pixi.toml
SCR=$ARU/src/scripts
LOG=/home/paperspace/logs
D=/home/paperspace/data/citrus_all/04_13D_Jackal
ROOTB=$D/blocks_ns/lio_row6F
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
FLOOR=25

until grep -q "CHAIN-04-ALL-DONE" $LOG/chain_04.log 2>/dev/null; do sleep 300; done
while ps -eo cmd | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 60; done
echo "=== FRUITFIX START $(date) ==="

for N in 001 000 002 003 004 005; do
  BD=$ROOTB/block_$N
  echo "=== FRUITFIX BLOCK $N ==="
  ( set -e
    cd $ARU
    [ -f "$D/sam3_fruit_tree_b$N/clip_000/frame_entries.json" ] || \
      pixi run --manifest-path $SAM3_PIXI python $SCR/fruit_in_trees_ledger.py \
        --data-dir $D --block-dir $BD --out-name sam3_fruit_tree_b$N
    python3 $SCR/compile_supervision.py --block-dir $BD \
      --tree-source idmap_dir --tree-src-dir $BD/supervision_trees_r6 \
      --hierarchy $D/scene_graph_v4/marker_hierarchy_fruit5.json \
      --fruit-ledger-glob "$D/sam3_fruit_tree_b$N/clip_*/frame_entries.json" \
      --out-dir $BD/supervision/strict_tree_v1 \
      | tee /tmp/fruitfix_compile_$N.txt
    KEPT=$(grep -oE "kept [0-9]+" /tmp/fruitfix_compile_$N.txt | grep -oE "[0-9]+" | head -1)
    echo "KEEPS block_$N: ${KEPT:-0}"
    if [ "${KEPT:-0}" -le "$FLOOR" ]; then
      echo "SKIP-TRAIN block_$N: keeps ${KEPT:-0} <= floor $FLOOR (fruit level not viable)"
      exit 0
    fi
    cd /home/paperspace/code/nerf_new
    echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
      LIDAR_DEPTH_NPZ=$BD/lidar_depth_morph.npz \
      pixi run ns-train high \
        --data $BD --output-dir $BD/splat_runs_S7 --experiment-name fruit_s7_b$N \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.output-depth-during-training True \
        --pipeline.model.high-loss-weight 1.0 \
        --pipeline.model.high-loss-fruit-weight 10.0 \
        --pipeline.model.fruit-protect True \
        --pipeline.model.sky-loss-lambda 1.0 \
        --pipeline.model.report-masked-metrics True \
        --pipeline.datamanager.semantic-dir $BD/supervision/strict_tree_v1 \
        --max-num-iterations 20001 --steps-per-save 9998 \
        --vis tensorboard nerfstudio-data \
        --eval-mode interval --eval-interval 10
    CFG=$(ls -t $BD/splat_runs_S7/fruit_s7_b$N/high/*/config.yml | head -1)
    sed "s|CFG = Path('.*')|CFG = Path('$CFG')|" $LOG/s2_gates_template.py > $LOG/s7_gates_b$N.py
    cd $ARU && HIGH_EMBEDDER_CKPT=$EMB \
      pixi run --manifest-path $NS_PIXI python $LOG/s7_gates_b$N.py 2>&1 \
      | grep -aE "RENDERED|fruit pixels|tree  pixels|CROSS-LEVEL|choosing" \
      | sed "s/^/GATES s7_b$N /"
    echo "FRUITFIX-DONE $N"
  ) || echo "FRUITFIX-FAIL $N"
done
echo "=== FRUITFIX-ALL-DONE $(date) ==="
