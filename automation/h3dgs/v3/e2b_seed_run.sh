#!/bin/bash
# e2b_seed_run.sh — the seeding ablation (e2b_seed_cell.sh: Z zero-init trained, R seed refined, P projection vote) over a list of identity cells
# (paper E2b, 2026-10-13). Default list: two cells per citrus survey + one per other site, chosen before any result was seen.
#   e2b_seed_run.sh [cells file (one identity dir per line)]   env: ARMS=Z,R,P STEPS=3000
set -uo pipefail; LIST=${1:-/home/paperspace/logs/e2b_cells.txt}; V3=/home/paperspace/code/automation/h3dgs/v3; L=/home/paperspace/logs/e2b_seed.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); say "=== E2b seeding arms ${ARMS:-Z,R,P} over $(grep -vc '^#' $LIST) cells"
grep -v '^#' $LIST | while read -r N; do [ -d "$N" ] && N=$N ARMS=${ARMS:-Z,R,P} STEPS=${STEPS:-3000} bash $V3/e2b_seed_cell.sh > /dev/null 2>&1 || say "missing $N"; done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
