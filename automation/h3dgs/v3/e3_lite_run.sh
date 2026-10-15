#!/bin/bash
# e3_lite_run.sh — LangSplat-lite (e3_lite_cell.sh) and the category protocol (tree=ALL, orange=FRUIT; LERF relevancy) for v3, flat96 and lite
# over a cell list (paper E3, 2026-10-13). Default list: the E2b cells.   e3_lite_run.sh [cells file]
set -uo pipefail; LIST=${1:-/home/paperspace/logs/e2b_cells.txt}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; L=/home/paperspace/logs/e3_lite.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); say "=== E3 lite + category over $(grep -vc '^#' $LIST) cells"
cat_score(){ [ -s $1/protocol_cat_$2.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $1 --tag cat_$2 --maps $3 "${@:4}" --category "tree=ALL,orange=FRUIT" 2>&1) | grep -aE '^\[protocol\].*category|Error|Traceback' | sed "s#^#[$(echo $1 | awk -F/ '{print $5"/"$7}')] #" | tee -a $L; }
grep -v '^#' $LIST | while read -r N; do
  [ -d "$N" ] || { say "missing $N"; continue; }; A=$N/arms
  N=$N bash $V3/e3_lite_cell.sh > /dev/null 2>&1
  cat_score $N v3 maps_v3/%s.npz --space hyper
  [ -e $A/features_flat96.bin.basis.npz ] && cat_score $N flat96 arms/maps_flat96/%s.npz --space flat --basis $A/features_flat96.bin.basis.npz
  [ -e $A/lite_targets/basis.npz ] && cat_score $N lite arms/maps_lite/%s.npz --space flat --basis $A/lite_targets/basis.npz --void-ref $A/features_lite.bin $N/features_B.bin
done; say "=== done in $(( ($(date +%s)-t0)/60 )) min"
