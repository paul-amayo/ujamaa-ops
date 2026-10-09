#!/bin/bash
# v3_lane_fleet.sh — lane chunk exports for A300 surveys beside the v3 fleet (2026-10-09), one lane at a time: v3_lane_chain.sh
# (extract -> SfM -> LO warp -> LiDAR init -> lane project) -> lane_heading_check.py (placed cameras vs the LiDAR odometry; Dec lane 2,
# the recipe of record, measures p90 0.98 / max 3.43 deg) -> if p90 <= 2.5 and max <= 6 deg: symlink chunks/lane<n> -> lane (a distinct
# chunk name, so the fleet's workspace and log names do not collide between lanes of one survey) and APPEND the training entry to the
# fleet queue ("<survey root> <lane>/h3dgs lane<n> 0"); otherwise HELD and logged. Logs: logs/v3_lane_fleet.log.
#   usage: v3_lane_fleet.sh <fleet queue> <survey root>:<lane> [<survey root>:<lane> ...]
set -u; Q=$1; shift; V3=/home/paperspace/code/automation/h3dgs/v3; A=/home/paperspace/code/automation/tenrows
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; L=/home/paperspace/logs/v3_lane_fleet.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== lane fleet: $# lanes -> queue $Q"
for item in "$@"; do
  S=${item%:*}; n=${item##*:}; LD=$S/experimental/lane$n; P=$LD/h3dgs; t0=$(date +%s)
  if grep -q " $P lane$n " $Q 2>/dev/null; then say "$(basename $S) lane $n: already queued"; continue; fi
  bash $V3/v3_lane_chain.sh $S $n chunk > /dev/null 2>&1
  if [ ! -e $P/camera_calibration/chunks/lane/sparse/0/images.bin ]; then say "$(basename $S) lane $n: chunk export FAILED (see logs/v3_lane_$(basename $S)_lane$n.log)"; continue; fi
  HC=$($PYH $A/lane_heading_check.py $LD 2>&1 | grep -a '^\[heading-check\]' | tail -1)
  p90=$(echo "$HC" | grep -oE 'p90 [0-9.]+' | head -1 | cut -d' ' -f2); mx=$(echo "$HC" | grep -oE 'max [0-9.]+' | head -1 | cut -d' ' -f2)
  init=$( [ -e $P/camera_calibration/chunks/lane/sparse/0/points3D.ply ] && echo yes || echo NO )
  say "$(basename $S) lane $n: export in $(( ($(date +%s)-t0)/60 )) min, LiDAR init $init; $HC"
  if [ "$init" = yes ] && python3 -c "import sys; sys.exit(0 if float('${p90:-99}') <= 2.5 and float('${mx:-99}') <= 6.0 else 1)"; then
    [ -e $P/camera_calibration/chunks/lane$n ] || ln -s lane $P/camera_calibration/chunks/lane$n
    echo "$S $P lane$n 0" >> $Q; say "$(basename $S) lane $n: QUEUED for v3 training (chunk lane$n)"
  else say "$(basename $S) lane $n: HELD (heading p90 ${p90:-?} max ${mx:-?} deg, init $init) - not queued"; fi
done
say "=== lane fleet done"
