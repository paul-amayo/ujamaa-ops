#!/bin/bash
# tenrows_lane_density.sh <lane dir> — is lane 2's 21.5 dB a recipe gain, or the training DENSITY the protocol gave it?
# (Paul, 2026-09-27: "did we isolate?" / "yeah" to the control.)
#
# The control originally proposed — score the h3dgs_lo3s SURVEY hierarchy on lane 2's held-out views — cannot be run:
# lo3s and every other ten_rows survey project were deleted in this morning's authorised cleanup (10:57 box, 27 G).
# The equivalent test without them is to make the LANE's training set as sparse as the survey's and re-measure.
#
# Measured on the lane's own poses: 562 frames over 46.3 m, consecutive spacing median 8.9 cm. The survey trained on
# keyframes cut at 20 cm or 3 deg, so:
#   --every 2  -> 281 training frames, ~18 cm apart  == the survey's density   <- the decisive arm
#   --every 5  -> 112 training frames, ~44 cm apart  (arm F as originally specified; the far end of the curve)
# Held-out stays the SAME 57 frames (every 10th; positions % 5 == 2 and % 2 == 0 offsets never collide with % 10 == 0),
# and the schedule/budget stay arm A's (60000 / 45000 / 0.0075 / 8 M), so density is the only variable.
# Arm A (all frames, 8.9 cm): held-out 21.54 median, training 25.73.
set -u
LD=${1:-/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/lane2}
L=/home/paperspace/logs/tenrows_lane_density.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
run(){ local proj=$1 every=$2; local t0=$(date +%s); local tag=${proj#h3dgs_}
  say "--- arm $proj: every ${every}th frame, arm A's schedule"
  env LANE_PROJ=$proj LANE_EVERY=$every bash /home/paperspace/logs/tenrows_lane_run.sh $LD 60000 45000 0.0075 8000000 \
    > /home/paperspace/logs/tenrows_lane2_${tag}_run.out 2>&1
  say "arm $proj finished in $(( ($(date +%s)-t0)/60 )) min: $(grep -aE 'lane-prep\] h3dgs|train rc|HELD-OUT|TRAINING|REFUSED|FAILED|NO ' /home/paperspace/logs/tenrows_lane_$(basename $LD)_${tag}.log 2>/dev/null | tail -5 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-190 | tr '\n' '|')"
}
say "queued: waiting for the fruit densification sweep and the 04 pipeline's last chunk"
while pgrep -f 'sidecar_fruit_densify\.sh' > /dev/null 2>&1 || pgrep -f 'train_single\.py' > /dev/null 2>&1; do sleep 120; done
say "=== lane density control on $(basename $LD); arm A (all frames, 8.9 cm) = 21.54 held-out / 25.73 training"
rm -rf $LD/h3dgs_e2 $LD/h3dgs_e5        # both are prep-only leftovers; the run rebuilds them
run h3dgs_e2 2
run h3dgs_e5 5
say "=== density control DONE"
