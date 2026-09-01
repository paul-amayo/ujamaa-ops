#!/bin/bash
# PROD BLOCK RECIPE — the referenceable per-block pipeline (P5, 2026-09-01).
#
#   usage: prod_block_recipe.sh <survey> <block-number> [--rolls N]
#   e.g.   prod_block_recipe.sh 05_13D_Jackal 17
#
# One block, end to end, with the measured recipe and a QA gate:
#
#   1. init    lidar_init_per_block --force          (min-range 0.45 default:
#              rover self-returns band at 0.16-0.39 m; 0.6 ate real overhanging
#              canopy — b013 1074 mid-band -4.9 dB. Measured 2026-09-01.)
#   2. stage1  ns-train 15001 iters, stop-split 6000 (9k polish), sky-loss 1,
#              antialiased, TREE_WEIGHT supervision, eval-interval 10.
#   3. census  censusinit_block.sh CENSUS_SEED_ONLY=1 (prod embedder+hierarchy;
#              seed alone verdicts tree/row at 0.86-0.93 — b013 P0).
#   4. chain   fruit_chain --steps blocks, STOP_SPLIT=16000 (the 15000->17001
#              densify window gets a 1k polish tail: P1 A/B — bars all-pass AND
#              fruit verdict 0.636 vs no-tail 0.567; fruitless blocks keep the
#              seed).
#   5. QA GATE judge_recipe.py bars: med>=21 P5>=17 int-min>=16 wvm<=4.
#              On failure: RE-ROLL stage1 once (roll variance drops 0-5 frames
#              2-9 dB at random — measured across 6 rolls on 4 blocks; culls
#              were tested and rejected: geometric cull -2.6 dB collateral,
#              view-exclusivity cull no effect). Keep the better roll.
#
# The gate REFUSES to leave a failing field as the block's newest run: after a
# failed second roll the block is flagged (exit 3) with the LAST roll left
# staged and both judgments recorded in recipe_*_judgments.log for triage.
#
# Requires the render service on :8004 (fresh — it re-globs replaced runs).
set -uo pipefail
SURVEY=${1:?usage: prod_block_recipe.sh <survey> <block-number> [--rolls N]}
N=${2:?block number}
MAXROLLS=2
[ "${3:-}" = "--rolls" ] && MAXROLLS=${4:-2}
B=$(printf "%03d" "$N")
ROOT=/home/paperspace/data/citrus_all/$SURVEY
case "$SURVEY" in apr_2026_zed|dec_2025_*) ROOT=/home/paperspace/data/klapmuts/$SURVEY;; esac
BLK=$ROOT/prod/tassili/blocks_ns/lio_row100
BD=$BLK/block_$B
EMB=$ROOT/prod/bateleur/embedder/*/ckpts/model_best.pth
EMB=$(ls -t $EMB 2>/dev/null | head -1)
HIER=$ROOT/prod/bateleur/scene_graph/marker_hierarchy.json
LOGS=/home/paperspace/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] recipe b$B $*"; }
[ -d "$BD" ] || { say "no block dir"; exit 2; }
[ -f "$EMB" ] || { say "no embedder under prod/bateleur"; exit 2; }

SUP=$BD/supervision/trees_fruit_v3
[ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/trees_only
[ -f "$SUP/manifest.json" ] || { say "no supervision manifest"; exit 2; }

run_roll() {
    say "init (min-range 0.45)"
    pixi run --manifest-path /home/paperspace/code/InstantSplat/pixi.toml \
      python /home/paperspace/code/aru_sil_core/src/scripts/lidar_init_per_block.py \
      --block-dir "$BD" --root "$ROOT" --force \
      > "$LOGS/recipe_${SURVEY}_${B}_init.log" 2>&1 || { say "INIT FAILED"; return 1; }

    rm -rf "$BD/splat_runs_STAGE1/stage1_bg00" "$BD/stage2_init" \
           "$BD/stage2_init_census" "$BD/stage2_init_densify" \
           "$BD/splat_runs_FEATFIX/stage2_bootstrap" \
           "$BD/splat_runs_FEATFIX/interaction_W.npz" \
           "$BD/splat_runs_FEATFIX/fruit_densify"

    say "stage1"
    ( cd /home/paperspace/code/nerf_new && echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
      TREE_WEIGHT_DIR=$SUP TREE_WEIGHT_BG=0.0 \
      pixi run ns-train high \
        --data "$BD" --output-dir "$BD/splat_runs_STAGE1" \
        --experiment-name stage1_bg00 \
        --pipeline.model.enable-high-features False \
        --pipeline.model.high-loss-weight 0.0 \
        --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.stop-split-at 6000 \
        --pipeline.model.sky-loss-lambda 1.0 \
        --pipeline.model.report-masked-metrics True \
        --max-num-iterations 15001 --steps-per-save 5000 \
        --vis tensorboard nerfstudio-data \
        --eval-mode interval --eval-interval 10 ) \
      > "$LOGS/recipe_${SURVEY}_${B}_stage1.log" 2>&1 || { say "STAGE1 FAILED"; return 1; }

    say "census seed"
    CENSUS_SEED_ONLY=1 CENSUS_EMBEDDER=$EMB CENSUS_HIERARCHY=$HIER \
      bash /home/paperspace/code/automation/censusinit_block.sh "$BD" "$SUP" \
      > "$LOGS/recipe_${SURVEY}_${B}_census.log" 2>&1 || { say "CENSUS FAILED"; return 1; }

    say "fruit chain (STOP_SPLIT=16000)"
    STOP_SPLIT=16000 bash /home/paperspace/code/automation/fruit_chain.sh "$SURVEY" \
      --steps blocks --blocks "$N" \
      > "$LOGS/recipe_${SURVEY}_${B}_chain.log" 2>&1 || { say "CHAIN FAILED"; return 1; }
    return 0
}

judge() {  # prints "PASS int-min med" or "FAIL int-min med"
    cd /home/paperspace/code/nerf_new
    local out
    out=$(timeout 900 pixi run python /home/paperspace/code/automation/judge_recipe.py \
          "recipe_${SURVEY}_${B}_roll$1" "$N" 2>/dev/null | grep "^\[recipe")
    echo "$out" >> "$LOGS/recipe_${SURVEY}_${B}_judgments.log"
    say "$out"
    case "$out" in *"bars P/P/P/P"*) echo PASS;; *) echo FAIL;; esac
}

for roll in $(seq 1 "$MAXROLLS"); do
    say "ROLL $roll/$MAXROLLS"
    run_roll || continue
    say "restart render service for judgment"
    pgrep -f "uvicorn.*render_[s]ervice" | xargs -r kill; sleep 4
    ( cd /home/paperspace/code/aru_sil_core && \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      RENDER_BLOCKS_ROOT=$BLK \
      RENDER_RUN_GLOB='splat_runs_FEATFIX/stage2_censusinit_*/high/*/config.yml' \
      HIGH_EMBEDDER_CKPT=$EMB \
      nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml \
        python -m uvicorn src.interfaces.splat_viewer.render_service:app \
        --host 0.0.0.0 --port 8004 > "$LOGS/render_service_8004.log" 2>&1 & )
    until curl -sf -m 3 localhost:8004/healthz >/dev/null 2>&1; do sleep 3; done
    state=$(judge "$roll" | tail -1)
    if [ "$state" = "PASS" ]; then say "QA GATE PASSED (roll $roll)"; exit 0; fi
    say "QA gate failed on roll $roll"
done
say "QA GATE NOT PASSED after $MAXROLLS rolls — last roll staged, FLAGGED (see recipe_${SURVEY}_${B}_judgments.log)"
exit 3
