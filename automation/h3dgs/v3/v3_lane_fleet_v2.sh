#!/bin/bash
# v3_lane_fleet_v2.sh — v3_lane_fleet.sh with the two CPU findings of 2026-10-09 applied: the chain runs through v3_lane_chain_v2.sh
# with MATCHER (default sequential, OVERLAP 30: a lane is one pass, no loop closures) and PINNED to cores 0-5 (taskset) so GLOMAP's
# 7-thread solve no longer starves the trainer's host thread (measured: GLOMAP at 706 % CPU on the 12-core box took the fleet's
# trainer from ~37 to 88 ms/it and GPU SM to 35-53 %). Same verdict rule: heading check p90 <= 2.5 / max <= 6 deg + LiDAR init.
#   usage: v3_lane_fleet_v2.sh <fleet queue> <survey root>:<lane> [...]      env: MATCHER=sequential|exhaustive, OVERLAP=30, CORES=0-5
set -u; Q=$1; shift; V3=/home/paperspace/code/automation/h3dgs/v3; A=/home/paperspace/code/automation/tenrows; CORES=${CORES:-0-5}
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; L=/home/paperspace/logs/v3_lane_fleet.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export MATCHER=${MATCHER:-sequential} OVERLAP=${OVERLAP:-30}
say "=== lane fleet v2: $# lanes -> queue $Q (matcher $MATCHER overlap $OVERLAP, cores $CORES)"
for item in "$@"; do
  S=${item%:*}; n=${item##*:}; LD=$S/experimental/lane$n; P=$LD/h3dgs; t0=$(date +%s)
  if grep -q " $P lane$n " $Q 2>/dev/null; then say "$(basename $S) lane $n: already queued"; continue; fi
  taskset -c $CORES bash $V3/v3_lane_chain_v2.sh $S $n chunk > /dev/null 2>&1
  if [ ! -e $P/camera_calibration/chunks/lane/sparse/0/images.bin ]; then say "$(basename $S) lane $n: chunk export FAILED (see logs/v3_lane_$(basename $S)_lane$n.log)"; continue; fi
  HC=$($PYH $A/lane_heading_check.py $LD 2>&1 | grep -a '^\[heading-check\]' | tail -1)
  p90=$(echo "$HC" | grep -oE 'p90 [0-9.]+' | head -1 | cut -d' ' -f2); mx=$(echo "$HC" | grep -oE 'max [0-9.]+' | head -1 | cut -d' ' -f2)
  init=$( [ -e $P/camera_calibration/chunks/lane/sparse/0/points3D.ply ] && echo yes || echo NO )
  say "$(basename $S) lane $n: export in $(( ($(date +%s)-t0)/60 )) min ($MATCHER), LiDAR init $init; $HC"
  if [ "$init" = yes ] && python3 -c "import sys; sys.exit(0 if float('${p90:-99}') <= 2.5 and float('${mx:-99}') <= 6.0 else 1)"; then
    [ -e $P/camera_calibration/chunks/lane$n ] || ln -s lane $P/camera_calibration/chunks/lane$n
    echo "$S $P lane$n 0" >> $Q; say "$(basename $S) lane $n: QUEUED for v3 training (chunk lane$n)"
  else say "$(basename $S) lane $n: HELD (heading p90 ${p90:-?} max ${mx:-?} deg, init $init) - not queued"; fi
done
say "=== lane fleet v2 done"
