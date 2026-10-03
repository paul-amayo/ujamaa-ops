#!/bin/bash
# block_rebuild.sh <block_dir> <paint|census> — one prod lio_row100 block after the 01 median-lifting swap (2026-10-03).
#  paint : quarantine the block's supervision-derived artifacts into <block>/era_liftmin_20261003/, then run_unified_pipeline
#          step 6a (save_filtered_semantic_pngs) + b0 (compile_supervision colour bridge, strict_fruit_tree_v1) verbatim
#  census: censusinit_block_glref.sh (re-census W on the new supervision; seed against the current embedder; stage the run)
set -uo pipefail
BD=$(readlink -f "$1"); STEP=${2:?paint|census}; N=$(basename $BD)
S=/home/paperspace/data/citrus_all/01_13B_Jackal; B=$S/prod/bateleur; MONOS=$S/prod/monos; HJ=$B/scene_graph/marker_hierarchy.json
ARU=/home/paperspace/code/aru_sil_core/src/scripts; ERA=$BD/era_liftmin_20261003; L=/home/paperspace/logs/liftmed_swap_01_blocks.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $N $STEP: $*" >> $L; }
cd /home/paperspace/code/nerf_new
case $STEP in
paint)
  mkdir -p $ERA
  for p in semantic_v2_B supervision/trees_only splat_runs_FEATFIX/interaction_W_glref.npz splat_runs_FEATFIX/stage2_provenance_glref.json splat_runs_FEATFIX/stage2_reseed_coral985.json; do
    [ -e $BD/$p ] && [ ! -e $ERA/$(echo $p | tr / _) ] && mv $BD/$p $ERA/$(echo $p | tr / _)
  done
  pixi run python $ARU/save_filtered_semantic_pngs.py --block-dir $BD --semantic-monolithic $MONOS/filtered_semantic_v2.monolithic \
     --marker-monolithic $B/scene_graph/markers_v2.monolithic --global-ids $B/sam3_v2/global_ids.json > /home/paperspace/logs/liftmed_01_${N}_paint.log 2>&1 \
     || { say "paint FAILED"; exit 1; }
  touch $BD/semantic_v2_B/.palette_v2
  pixi run python $ARU/compile_supervision.py --block-dir $BD --tree-source colour_png_bridge --hierarchy $HJ \
     --fruit-ledger-glob "$B/sam3_fruit/clip_*/frame_entries.json" --filter strict_fruit_tree_v1 --out-dir $BD/supervision/trees_only \
     > /home/paperspace/logs/liftmed_01_${N}_compile.log 2>&1 || { say "compile FAILED"; exit 1; }
  [ -f $BD/supervision/trees_only/manifest.json ] && say "ok ($(ls $BD/supervision/trees_only/kf_*.png 2>/dev/null | wc -l) maps)" || { say "no manifest"; exit 1; } ;;
census)
  RUN=$(ls -d $BD/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/ 2>/dev/null | head -1)
  mkdir -p $ERA
  for c in $RUN/nerfstudio_models/*.ckpt; do [ -e "$c" ] && [ ! -e $ERA/seed_$(basename $c) ] && mv "$c" $ERA/seed_$(basename $c); done
  [ -e $BD/splat_runs_FEATFIX/interaction_W_glref.npz ] && mv $BD/splat_runs_FEATFIX/interaction_W_glref.npz $ERA/interaction_W_glref.npz.census2
  CENSUS_EMBEDDER=$B/embedder/01_13B_v1g/ckpts/model_best.pth CENSUS_HIERARCHY=$HJ \
    bash /home/paperspace/code/automation/citrus_glref/censusinit_block_glref.sh $BD > /home/paperspace/logs/liftmed_01_${N}_census.log 2>&1
  ls $RUN/nerfstudio_models/*.ckpt > /dev/null 2>&1 && grep -q "REPL-SEEDONLY" /home/paperspace/logs/liftmed_01_${N}_census.log \
    && say "ok ($(grep -aoE 'assigned [0-9]+/[0-9]+' /home/paperspace/logs/liftmed_01_${N}_census.log | tail -1))" || { say "FAILED (see liftmed_01_${N}_census.log)"; exit 1; } ;;
esac
