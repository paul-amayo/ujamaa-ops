#!/bin/bash
# The loop over every citrus 05 fruit3d tree with a given detector checkpoint: track (fine-tuned image weights inside the video
# model) -> tri -> score into fewshot_<tag>/; compare confirmed-orange counts with the stock fleet (fewshot/). usage: fleet_loop.sh <tag> <trainer ckpt>
set -u; TAG=$1; CK=$2; S=/home/paperspace/code/automation/sam3_fewshot/pilot_t72.py; PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python; L=/home/paperspace/logs/sam3_fewshot_loop_$TAG.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; export POSES=refined KF_ONLY=1 REPROJ_PX=6 FEWSHOT_DIR=fewshot_$TAG SAM3_DET_CKPT=$CK; t0=$(date +%s); n=0; cd /home/paperspace/code/sam3
for W in /home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/fruit3d_t*/; do T=$(basename $W); [ -e $W/detections.json ] || continue; n=$((n+1))
  [ -e $W/fewshot_$TAG/score_refined_kfonly_thr6.json ] && { say "$T: done already"; continue; }
  FRUIT3D_WORK=$W timeout 1200 $PY $S track 2>&1 | grep -a '^\[track\] [0-9]* tracks\|Error\|Traceback' | sed "s/^/[$T] /" | tee -a $L
  FRUIT3D_WORK=$W $PY $S tri 2>&1 | grep -a '^\[tri\] [0-9]* tracks\|Error' | sed "s/^/[$T] /" | tee -a $L; FRUIT3D_WORK=$W $PY $S score 2>&1 | grep -a '^\[score\]' | cut -c1-160 | sed "s/^/[$T] /" | tee -a $L
done; say "=== loop $TAG done: $n trees in $(( ($(date +%s)-t0)/60 )) min"
