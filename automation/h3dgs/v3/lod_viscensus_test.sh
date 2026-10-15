#!/bin/bash
# lod_viscensus_test.sh — the visibility-aware census test (2026-10-14): for one splat geometry, re-run the census with --visible-only, then re-seed
# with the 10 % / 30 % / majority fruit rules (lod_majority_test.sh), render the fruit frames and score E4 — compare with the same rules on the
# ordinary census.   CFG=<run config.yml> SRC=<identity dir of that geometry> OUT=<prefix for the new identity dirs> TAGP=<tag prefix> SUP=<id maps> FIGDIR=<dir>
set -uo pipefail; : ${CFG:?} ${SRC:?} ${OUT:?} ${TAGP:?} ${SUP:?}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/lod_viscensus_$TAGP.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $TAGP: $*" | tee -a $L; }; t0=$(date +%s)
VW=${OUT}_vis/W.npz; mkdir -p ${OUT}_vis
[ -s $VW ] || (cd $NS && nice -n 5 pixi run python $V3/splat_census.py --config $CFG --supervision-dir $SUP --names-json $SRC/split_names.json --with-bg --vis-weight --out-npz $VW 2>&1) | grep -aE '^\[census\] saved|Error|Traceback' | cut -c1-400 | tee -a $L
[ -s $VW ] || { say "visible census FAILED"; exit 1; }
for SH in 0.1 0.3 0; do t=$(echo $SH | tr -d .); D=${OUT}_vis_s$t; mkdir -p $D; cp -n $VW ${OUT}_vis/W.json $D/ 2>/dev/null
  CFG=$CFG SRC=$SRC DST=$D TAG=${TAGP}vis_s$t SHARE=$SH SUP=$SUP FIGDIR=${FIGDIR:-$SRC/figs/lod} bash $V3/lod_majority_test.sh > /dev/null 2>&1
  say "rule $SH: $(grep -a 'fruit-word gaussians' /home/paperspace/logs/lod_majority_${TAGP}vis_s$t.log | tail -1)"; done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
