#!/bin/bash
# tenrows_lane_ablation_v2.sh <lane dir> — what raised lane 2's render quality (Paul, 2026-09-27: "find exactly what raised
# the render quality, is it more frames, densification" / "we know 30k works" / "ablation should be smaller densification
# budget, and then every fifth frame"). Arm A = the proven run (project h3dgs: all frames, 60k / densify to 45k / grad
# 0.0075 / 8 M budget → held-out 21.54, training 25.73). Same lane, same warped poses, same LiDAR init, same 57 held-out
# frames (every 10th); sequential on the one GPU:
#   E  h3dgs_b4m   all frames, same schedule, budget 4 M (half)           -> is it the gaussian count (densification)?
#   F  h3dgs_e5    every 5th frame (positions % 5 == 2), same schedule, 8 M -> is it the frame count?
set -u
LD=${1:?lane dir}; L=/home/paperspace/logs/tenrows_lane_ablation.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
run(){ # <proj> <env assignments...> -- <iters> <du> <grad> <cap>
  local proj=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local t0=$(date +%s); local tag=${proj#h3dgs_}
  say "--- arm $proj (${envs[*]:-all frames}) schedule $*"
  env LANE_PROJ=$proj "${envs[@]}" bash /home/paperspace/logs/tenrows_lane_run.sh $LD "$@" > /home/paperspace/logs/tenrows_lane2_${tag}_run.out 2>&1
  say "arm $proj finished in $(( ($(date +%s)-t0)/60 )) min: $(grep -aE 'lane-prep\] h3dgs|train rc|HELD-OUT|TRAINING|REFUSED|FAILED|NO ' /home/paperspace/logs/tenrows_lane_$(basename $LD)_${tag}.log | tail -5 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-190 | tr '\n' '|')"
}
say "=== ablation v2 on $(basename $LD): E (budget 4 M, all frames) then F (every 5th frame, 8 M); arm A = h3dgs (done)"
run h3dgs_b4m -- 60000 45000 0.0075 4000000
run h3dgs_e5 LANE_EVERY=5 -- 60000 45000 0.0075 8000000
say "=== ablation v2 DONE"
