#!/bin/bash
# tenrows_lane_ablation_v3.sh <lane dir> — arm G (Paul, 2026-09-27: "Arm G for 30k"): all frames with the survey's known
# schedule (30k iterations, densify to 15k, grad 0.015, no budget) — the recipe of record on the lane's full stream, so
# G vs A isolates the long budgeted schedule and G vs the survey chunks isolates the frame count / single-lane setup.
# Waits for ablation v2 (arms E, F) to finish, then runs G through the same lane driver; same log, same monitor.
set -u
LD=${1:?lane dir}; L=/home/paperspace/logs/tenrows_lane_ablation.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until grep -q 'ablation v2 DONE' $L; do sleep 60; done
proj=h3dgs_full30; tag=full30; t0=$(date +%s)
say "--- arm $proj (all frames, survey schedule) schedule 30000 15000 0.015 0"
env LANE_PROJ=$proj bash /home/paperspace/logs/tenrows_lane_run.sh $LD 30000 15000 0.015 0 > /home/paperspace/logs/tenrows_lane2_${tag}_run.out 2>&1
say "arm $proj finished in $(( ($(date +%s)-t0)/60 )) min: $(grep -aE 'lane-prep\] h3dgs|train rc|HELD-OUT|TRAINING|REFUSED|FAILED|NO ' /home/paperspace/logs/tenrows_lane_$(basename $LD)_${tag}.log | tail -5 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-190 | tr '\n' '|')"
say "=== ablation v3 (arm G) DONE"
