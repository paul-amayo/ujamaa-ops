#!/bin/bash
# v3_supervision_prebuild.sh — build the chunk supervision (v3_chunk_supervision.sh) for queued cells AHEAD of the v3 fleet, on CPU
# (2026-10-09): written to the cell dir the fleet looks for (<survey root>/experimental/v3/chunk_<cn>/cell), so when the fleet reaches
# the cell it finds the manifest and goes straight to the census; failures show up hours before the fleet gets there. Surveys without
# SAM3 fruit ledgers (02, 03) compile trees + rows only (an empty fruit glob is accepted by compile_supervision).
#   usage: v3_supervision_prebuild.sh <survey id under citrus_all> <H3DGS project> <chunk> [<chunk> ...]
set -u; SVN=$1; PROJ=$2; shift 2; V3=/home/paperspace/code/automation/h3dgs/v3; L=/home/paperspace/logs/v3_supervision_prebuild.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; S=/home/paperspace/data/citrus_all/$SVN
for CN in "$@"; do
  OUT=$S/experimental/v3/chunk_$CN/cell; t0=$(date +%s)
  [ -e $OUT/supervision/trees_only/manifest.json ] && [ -e $OUT/frames.json ] && { say "$SVN $CN: already built"; continue; }
  SVN=$SVN PROJ=$PROJ CN=$CN OUT=$OUT bash $V3/v3_chunk_supervision.sh > /dev/null 2>&1
  if [ -e $OUT/supervision/trees_only/manifest.json ]; then
    say "$SVN $CN: $(ls $OUT/supervision/trees_only | grep -c png) maps, $(python3 -c "import json; m=json.load(open('$OUT/supervision/trees_only/manifest.json')); wt=m.get('word_table', {}); print(sum(1 for k in wt if int(k) < 10000), 'tree ids,', sum(1 for k in wt if int(k) >= 10000), 'fruit ids')") in $(( $(date +%s)-t0 )) s"
  else say "$SVN $CN: FAILED (see logs/v3_supervision_${SVN}_${CN}.log*)"; fi
done
