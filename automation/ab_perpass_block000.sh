#!/bin/bash
# A/B: per-pass COLMAP-refined poses vs raw LIO poses on 05 lio_row100
# block_000 (Paul 2026-08-13). Same stage1_bg00 flags, same init, same
# frames — the ONLY difference is the pose source. The May perpass fleet
# suggests refined poses are worth several dB of train FG; 100-kf blocks
# are single-pass so the refine is well-posed and the Umeyama-to-LIO
# anchor keeps everything metric.
set -uo pipefail
ROOT=/home/paperspace/data/citrus_all/05_13D_Jackal
BD=$ROOT/blocks_ns/lio_row100/block_000
BREF=$ROOT/blocks_ns/lio_row100/block_000_ppref
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
LOG=/home/paperspace/logs/ab_perpass_block000.log
SKILL_READ=/home/paperspace/.claude/skills/psnr/scripts/read_tb_scalars.py

echo "=== A/B perpass block_000 $(date) ==="

# 0. wait for the raw-LIO stage1 (arm A) to finish
until ls "$BD"/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; do
    sleep 60
done
echo "arm A (raw LIO) ckpt present"

# 1. per-pass refine (single pass for a row block)
if [ ! -f "$BD/transforms_refined_per_pass.json" ]; then
    pixi run --manifest-path "$NS_PIXI" python "$SRC/per_pass_colmap_refine.py" \
        --block-dir "$BD" --root "$ROOT" \
        --log-dir "$ROOT/_logs" \
        || { echo "AB-FAIL refine"; exit 1; }
fi
echo "refine done"

# 2. arm B block dir: refined transforms + same lidar init
mkdir -p "$BREF"
cp "$BD/transforms_refined_per_pass.json" "$BREF/transforms.json"
cp "$BD/init_lidar.ply" "$BREF/init_lidar.ply"
echo "arm B dir built ($BREF)"

# 3. arm B stage1 — flags identical to the pipeline's stage1_bg00
if ! ls "$BREF"/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
    cd /home/paperspace/code/nerf_new
    echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
      TREE_WEIGHT_DIR= TREE_WEIGHT_BG=0.0 \
      pixi run ns-train high \
        --data "$BREF" --output-dir "$BREF/splat_runs_STAGE1" \
        --experiment-name stage1_bg00 \
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
        || { echo "AB-FAIL arm B train"; exit 1; }
fi
echo "arm B trained"

# 4. verdict: FG curves side by side
for ARM in "A:$BD" "B:$BREF"; do
    NAME=${ARM%%:*}; DIR=${ARM#*:}
    echo "--- arm $NAME ($DIR)"
    pixi run --manifest-path "$NS_PIXI" python "$SKILL_READ" \
        "$DIR/splat_runs_STAGE1/stage1_bg00" \
        --tag-substring psnr_fg --milestones 2000 5000 10000 15000
done
echo "=== A/B COMPLETE $(date) ==="
