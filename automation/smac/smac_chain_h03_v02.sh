#!/usr/bin/env bash
# smac_chain_h03_v02.sh (1 Nov 2026) — detached: finish the h03 arms (raw, descriptors; upstream MapClosures constants), then the v02 set
# (all three arms, keyframe voxel 0.2 m, loosened MapClosures ORB/self-similarity via SMAC_MC_* env), then score every arm.
set -u
A=/home/paperspace/code/automation; D=/mnt/shared2/smac_citrus; R=$D/runs; T=$D/truth
T5=$T/05_13D_Jackal; T4=$T/04_13D_Jackal; S5=$D/05_13D_Jackal; S4=$D/04_13D_Jackal
echo "[$(date "+%F %T")] chain start pid $$"
bash $A/smac_run_arms.sh h03 7528 7626 -0.2 0.2 0.5 "raw descriptors"
export SMAC_MC_SELFSIM=0 SMAC_MC_FAST=10 SMAC_MC_EDGE=15 SMAC_MC_PATCH=15 SMAC_MC_NLEVELS=4 SMAC_MC_SCALE=1.2 SMAC_MC_NFEAT=1000
echo "[$(date "+%F %T")] v02: voxel 0.2, MC env: SELFSIM=$SMAC_MC_SELFSIM FAST=$SMAC_MC_FAST EDGE=$SMAC_MC_EDGE PATCH=$SMAC_MC_PATCH NLEVELS=$SMAC_MC_NLEVELS SCALE=$SMAC_MC_SCALE NFEAT=$SMAC_MC_NFEAT"
bash $A/smac_run_arms.sh v02 7528 7626 -0.2 0.2 0.2 "segments raw descriptors"
for run in h03_segments h03_raw v02_segments v02_raw; do
  echo "===== closures $run"; grep -c "stack trace" $R/$run/run.log; grep -h "best closure\|repeat keyframe" $R/$run/run.log | grep -c "teach-referenced"
  python3 $A/smac_eval_closures.py $R/$run $S5 $S4 $T5 $T4 2>&1 | tail -15
done
for run in h03_descriptors v02_descriptors; do
  echo "===== descriptors $run"; python3 $A/smac_eval_descriptors.py $R/$run $S5 $S4 $T5 $T4 2>&1 | tail -15
done
echo "[$(date "+%F %T")] CHAIN DONE"
