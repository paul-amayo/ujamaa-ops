#!/bin/bash
# Rebuild + export the tree-72 story blocks of 05 for the splat viewer.
#
# The fruit chain's finals (densify -> census on densified -> share-seed init,
# staged as stage2_censusinit_seed) were verdicted then RECLAIMED for disk.
# This replays exactly that per-block recipe (fruit_chain.sh lines 190-227)
# for the demo set, then runs export_register_stage2.sh (ns-export -> crop ->
# world-frame ply + features.npy -> back-attribution -> merge into
# splats.json) so the viewer serves gen2 fruit-aware splats.
#   usage: demo_stage_05.sh [block ...]   (default: 013 014 027)
set -u
export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
R=/home/paperspace/data/citrus_all/05_13D_Jackal
CFGDIR=$R/prod/tassili/blocks_ns/lio_row100
EMB=$R/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth
SCR=/home/paperspace/code/aru_sil_core/src/scripts
LOGS=/home/paperspace/logs/demo_stage_05
mkdir -p "$LOGS"
mark() { echo "[$(date +%F_%T)] $*" | tee -a "$LOGS/STATUS"; }

BLOCKS=("$@"); [ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=(013 014 027)

for N in "${BLOCKS[@]}"; do
  BD=$CFGDIR/block_$N
  [ -d "$BD" ] || { mark "b$N missing"; continue; }
  if ls "$BD"/splat_runs_FEATFIX/stage2_censusinit_seed/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
    mark "b$N seed cache hit"
  else
    mark "b$N densify"
    rm -rf "$BD/splat_runs_FEATFIX/fruit_densify" "$BD/stage2_init_densify" \
           "$BD/splat_runs_FEATFIX/interaction_W_densified.npz"
    CENSUS_EMBEDDER=$EMB bash /home/paperspace/code/automation/densify_block.sh \
        "$BD" "$BD/supervision/trees_fruit_v3" 2000 \
        > "$LOGS/b${N}_densify.log" 2>&1 || { mark "b$N densify FAILED"; continue; }
    DRUN=$(ls -dt "$BD"/splat_runs_FEATFIX/fruit_densify/high/*/ | head -1)
    DCK=$(ls -t "$DRUN"/nerfstudio_models*/*.ckpt | head -1)
    WD=$BD/splat_runs_FEATFIX/interaction_W_densified.npz
    mark "b$N census"
    (cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
        "$SCR/gaussian_interaction_census.py" \
        --run-glob "$DRUN/config.yml" \
        --supervision-dir "$BD/supervision/trees_fruit_v3" --out-npz "$WD") \
        > "$LOGS/b${N}_census.log" 2>&1 || { mark "b$N census FAILED"; continue; }
    (cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
        "$SCR/build_census_init.py" \
        --w-npz "$WD" --embedder "$EMB" --src-ckpt "$DCK" \
        --dst-dir "$BD/stage2_init_densify/nerfstudio_models" \
        --fruit-share-assign 0.1) \
        >> "$LOGS/b${N}_census.log" 2>&1 || { mark "b$N share-seed FAILED"; continue; }
    TS=$(basename "$DRUN")
    RUN=$BD/splat_runs_FEATFIX/stage2_censusinit_seed/high/$TS
    rm -rf "$BD/splat_runs_FEATFIX/stage2_censusinit_seed" \
           "$BD/splat_runs_FEATFIX/stage2_censusinit_fw2"
    mkdir -p "$RUN/nerfstudio_models"
    sed 's|^experiment_name: .*$|experiment_name: stage2_censusinit_seed|' \
        "$DRUN/config.yml" > "$RUN/config.yml"
    cp "$DRUN/dataparser_transforms.json" "$RUN/" 2>/dev/null
    cp "$BD"/stage2_init_densify/nerfstudio_models/*.ckpt "$RUN/nerfstudio_models/"
    mark "b$N staged stage2_censusinit_seed"
  fi
  mark "b$N export"
  bash /home/paperspace/code/automation/export_register_stage2.sh "$BD" \
      > "$LOGS/b${N}_export.log" 2>&1 \
      && mark "b$N EXPORT DONE" || mark "b$N export FAILED (see b${N}_export.log)"
done
mark "DEMO-STAGE-DONE"
