#!/bin/bash
# Fleet reseed with the LiDAR self-return filter (Paul, 2026-09-01).
#
# For every listed block: regenerate init_lidar.ply (min-range 0.6 m to the
# camera path — lidar_init_per_block.py default since 2026-08-31), retrain
# stage1 from the filtered cloud, re-stage the census-init seed. Then ONE
# fruit_chain --steps blocks pass over the list (densify+train+verdict for
# fruit-bearing blocks; fruitless blocks keep the freshly staged seed).
#
# Measured on b040/b013: init 1-2.5 min, stage1 6.5 min, census 1.5 min,
# fruit-chain step ~14 min/fruit block. A/B that justified this: b040
# kf_003255 13.64 -> 22.65 dB, fruit verdict 0.825 -> 0.962.
#
# Old stage1 runs are DELETED (not sidelined): b040/b013 keep _selfret
# sidelines as the recorded A/B pair; carrying 41 more superseded runs
# (~0.5-1 GB each) against the 15 G disk floor serves nothing.
set -uo pipefail
ROOT=/home/paperspace/data/citrus_all/05_13D_Jackal
BLK=$ROOT/prod/tassili/blocks_ns/lio_row100
GATE=${GATE:-/home/paperspace/logs/.b13_reshoot_done}
BLOCKS=${BLOCKS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 41 42"}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
floor_ok() {
    local a; a=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$a" -ge 15 ] || { say "ABORT-DISK: ${a}G < 15G floor"; return 1; }
}

say "fleet reseed queued — waiting for gate $GATE"
until [ -f "$GATE" ]; do sleep 60; done
say "gate open — starting"

FAILED=""
for n in $BLOCKS; do
    B=$(printf "%03d" "$n"); BD=$BLK/block_$B
    [ -d "$BD" ] || { say "SKIP b$B: no dir"; continue; }
    floor_ok || exit 1
    say "b$B init (filtered)"
    pixi run --manifest-path /home/paperspace/code/InstantSplat/pixi.toml \
      python /home/paperspace/code/aru_sil_core/src/scripts/lidar_init_per_block.py \
      --block-dir "$BD" --root "$ROOT" --force \
      > /home/paperspace/logs/reseed_${B}_init.log 2>&1 \
      || { say "b$B INIT FAILED"; FAILED="$FAILED $B"; continue; }
    grep -h "min-range" /home/paperspace/logs/reseed_${B}_init.log | tail -1

    rm -rf "$BD/splat_runs_STAGE1/stage1_bg00" "$BD/stage2_init" \
           "$BD/stage2_init_census" "$BD/stage2_init_densify" \
           "$BD/splat_runs_FEATFIX/stage2_bootstrap" \
           "$BD/splat_runs_FEATFIX/interaction_W.npz" \
           "$BD/splat_runs_FEATFIX/fruit_densify"

    say "b$B stage1"
    SUP=$BD/supervision/trees_fruit_v3
    [ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/trees_only
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
      > /home/paperspace/logs/reseed_${B}_stage1.log 2>&1 \
      || { say "b$B STAGE1 FAILED"; FAILED="$FAILED $B"; continue; }

    say "b$B census seed"
    CENSUS_SEED_ONLY=1 \
    CENSUS_EMBEDDER=$ROOT/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth \
    CENSUS_HIERARCHY=$ROOT/prod/bateleur/scene_graph/marker_hierarchy.json \
      bash /home/paperspace/code/automation/censusinit_block.sh "$BD" "$SUP" \
      > /home/paperspace/logs/reseed_${B}_census.log 2>&1 \
      || { say "b$B CENSUS FAILED"; FAILED="$FAILED $B"; continue; }
    say "b$B reseeded"
done
say "seed pass done; failed:${FAILED:- none}"

GOOD=$(for n in $BLOCKS; do case " $FAILED " in *" $(printf %03d $n) "*) ;; *) printf "%s," "$n";; esac; done)
say "fruit chain over ${GOOD%,}"
bash /home/paperspace/code/automation/fruit_chain.sh 05_13D_Jackal \
  --steps blocks --blocks "${GOOD%,}" \
  > /home/paperspace/logs/fleet_reseed_fruitchain.log 2>&1 \
  || say "FRUIT CHAIN exited non-zero"
say "FLEET RESEED DONE (failed:${FAILED:- none})"
