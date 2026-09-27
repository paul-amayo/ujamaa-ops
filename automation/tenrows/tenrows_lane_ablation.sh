#!/bin/bash
# tenrows_lane_ablation.sh <lane dir> — what raised lane 2's render quality (Paul, 2026-09-27: "find exactly what raised the
# render quality, is it more frames, densification")? 2 x 2 on the SAME lane, poses (lane SfM warped onto the LiDAR
# odometry), LiDAR init and held-out set (every 10th 15 Hz frame) — only the training-frame set and the training schedule
# change. Arm A = the proven run (project h3dgs: all frames, 60k / densify to 45k / grad 0.0075 / 8 M budget). Sequential
# on the one GPU, cheapest first:
#   C  h3dgs_full30  all frames      + survey schedule 30k / 15k / 0.015, no budget   -> isolates the schedule
#   D  h3dgs_kf30    keyframes only  + survey schedule                                -> the survey recipe on one lane
#   B  h3dgs_kf60    keyframes only  + long budgeted schedule                         -> isolates the frame count
set -u
LD=${1:?lane dir}; L=/home/paperspace/logs/tenrows_lane_ablation.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
run(){ # <proj> <kf 0|1> <iters> <du> <grad> <cap>
  local proj=$1 kf=$2; shift 2; local t0=$(date +%s); local tag=${proj#h3dgs_}
  say "--- arm $proj (keyframes=$kf) schedule $*"
  if [ "$kf" = 1 ]; then LANE_PROJ=$proj LANE_KF=1 bash /home/paperspace/logs/tenrows_lane_run.sh $LD "$@" > /home/paperspace/logs/tenrows_lane2_${tag}_run.out 2>&1
  else LANE_PROJ=$proj bash /home/paperspace/logs/tenrows_lane_run.sh $LD "$@" > /home/paperspace/logs/tenrows_lane2_${tag}_run.out 2>&1; fi
  say "arm $proj finished in $(( ($(date +%s)-t0)/60 )) min: $(grep -aE 'lane-prep\] h3dgs|train rc|HELD-OUT|TRAINING|REFUSED|FAILED|NO ' /home/paperspace/logs/tenrows_lane_$(basename $LD)_${tag}.log | tail -5 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-190 | tr '\n' '|')"
}
say "=== ablation on $(basename $LD): arms C (full30), D (kf30), B (kf60); arm A = h3dgs (done)"
run h3dgs_full30 0 30000 15000 0.015 0
run h3dgs_kf30   1 30000 15000 0.015 0
run h3dgs_kf60   1 60000 45000 0.0075 8000000
say "=== ablation DONE"
