#!/bin/bash
# SAM3 few-shot recipe over every citrus 05 fruit3d tree (Paul 2026-10-07: "go ahead with the recipe"): per tree, pilot_t72.py
# track (SAM3 video, stock-level thresholds) -> tri (keyframe views, trained poses) -> score (stock recall on confirmed oranges)
# -> exemplar (K = 1/3/5). One tree per python call so a failure stays local; fleet_summary.py aggregates.
set -u
S=/home/paperspace/code/automation/sam3_fewshot/pilot_t72.py; PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python; L=/home/paperspace/logs/sam3_fewshot_fleet_05.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; export POSES=refined KF_ONLY=1 REPROJ_PX=6; t0=$(date +%s); n=0
for W in $(ls -d /home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/fruit3d_t*/ | sort -t_ -k2.2 -n); do
  [ -e $W/detections.json ] && [ -e $W/meta.json ] || continue; T=$(basename $W); n=$((n+1)); t1=$(date +%s)
  if [ -e $W/fewshot/exemplar_refined_kfonly_thr6.json ]; then say "$T: done already"; continue; fi
  cd /home/paperspace/code/sam3
  { [ -e $W/fewshot/tracks.json ] && [ "$T" = fruit3d_t72 ] || FRUIT3D_WORK=$W timeout 900 $PY $S track; } 2>&1 | grep -a '^\[track\] [0-9]* tracks\|Error\|Traceback' | sed "s/^/[$T] /" | tee -a $L
  [ -e $W/fewshot/tracks.json ] || { say "$T: NO TRACKS"; continue; }
  FRUIT3D_WORK=$W $PY $S tri 2>&1 | grep -a '^\[tri\] [0-9]* tracks\|Error\|Traceback' | sed "s/^/[$T] /" | tee -a $L
  FRUIT3D_WORK=$W $PY $S score 2>&1 | grep -a '^\[score\]\|Error\|Traceback' | sed "s/^/[$T] /" | tee -a $L
  FRUIT3D_WORK=$W timeout 1500 $PY $S exemplar 2>&1 | grep -a '^\[exemplar\]\|Error\|Traceback' | sed "s/^/[$T] /" | tee -a $L
  say "$T done in $(( $(date +%s)-t1 )) s ($n trees, $(( ($(date +%s)-t0)/60 )) min)"
done
say "=== fleet done: $n trees in $(( ($(date +%s)-t0)/60 )) min"
