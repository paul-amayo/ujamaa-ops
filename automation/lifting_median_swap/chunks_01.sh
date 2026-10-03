#!/bin/bash
# 01 chunk cells after the median-lifting swap (Paul 2026-10-03: "why are we doing block supervision? we need to go straight
# to chunks"). Per chunk with a trained hierarchy.hier_opt: fresh chunk_<CN>_sam3 cell via native_sam3_cell.sh (chunk
# supervision painted from the new filtered semantic monolithic, census on the chunk's train views, seeds A/B, scoring), then
# the text bank. No side-car reference seed (deprecated). An existing cell moves to <cell>_era_liftmin_20261003.
set -uo pipefail
S=/home/paperspace/data/citrus_all/01_13B_Jackal; P=$S/experimental/h3dgs; NAT=/home/paperspace/code/automation/h3dgs/native
HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; L=/home/paperspace/logs/liftmed_chunks_01.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
for CN in ${*:-3_1 2_0 2_2 6_2 7_2}; do
  [ -e $P/output/trained_chunks/$CN/hierarchy.hier_opt ] || { say "chunk $CN: no hierarchy.hier_opt — skipped"; continue; }
  OUT=$S/experimental/h3dgs_native/chunk_${CN}_sam3
  [ -d $OUT ] && [ ! -d ${OUT}_era_liftmin_20261003 ] && mv $OUT ${OUT}_era_liftmin_20261003
  MUST=""; [ "$CN" = "3_1" ] && MUST=kf_003426.png
  say "chunk $CN: native_sam3_cell -> $OUT"
  SVN=01_13B_Jackal PROJ=$P CN=$CN OUT=$OUT MUST=$MUST bash $NAT/native_sam3_cell.sh > /home/paperspace/logs/liftmed_chunk_01_$CN.log 2>&1
  if [ -f $OUT/features_B_bg2share.bin ] && [ -f $OUT/supervision/trees_only/manifest.json ]; then
    (cd /home/paperspace/code/nerf_new && pixi run python $NAT/build_text_bank.py --manifest $OUT/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $OUT/text_bank.npz 2>&1 | grep -a "text-bank\|Error" | tee -a $L)
    say "chunk $CN: done ($(grep -aE 'assigned [0-9]+/' $OUT/features_B_bg2share.bin.json 2>/dev/null; python3 -c "import json; d=json.load(open('$OUT/features_B_bg2share.bin.json')); print(d['assigned'], 'of', d['n'], 'nodes seeded')"))"
  else say "chunk $CN: FAILED (see liftmed_chunk_01_$CN.log)"; fi
done
say "=== chunks done"
