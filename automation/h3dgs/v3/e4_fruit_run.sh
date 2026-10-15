#!/bin/bash
# e4_fruit_run.sh — e4_fruit_cell.sh over the fruit-bearing cells (04 and 05, 2026-10-13).   e4_fruit_run.sh [cells file]
set -uo pipefail; LIST=${1:-/home/paperspace/logs/e4_cells.txt}; V3=/home/paperspace/code/automation/h3dgs/v3; L=/home/paperspace/logs/e4_fruit.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); say "=== E4 fruit frames over $(grep -vc '^#' $LIST) cells"
grep -v '^#' $LIST | while read -r N; do [ -d "$N" ] && N=$N bash $V3/e4_fruit_cell.sh > /dev/null 2>&1 || say "missing $N"; done; say "=== done in $(( ($(date +%s)-t0)/60 )) min"
