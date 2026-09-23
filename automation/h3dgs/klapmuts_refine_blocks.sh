#!/bin/bash
# Per-block pose refinement for a Klapmuts survey with the citrus glref recipe of record: flip untagged OpenCV-era
# blocks to OpenGL (byte backup + tag), then per-block GPU SIFT + exhaustive match + GLOMAP + Sim(3) into the LIO
# world in place (guards: coverage >=90 %, p50 <=0.5 m; odometry backup transforms_odo_glfix.json).
SV=${1:?survey root}; CFG=$SV/prod/tassili/blocks_ns/lio_row100; LOG=/home/paperspace/logs/klapmuts_refine_$(basename $SV).log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
say "=== refine $(basename $SV): $(ls -d $CFG/block_[0-9][0-9][0-9] | wc -l) canonical blocks"
for BD in $CFG/block_[0-9][0-9][0-9]; do
  B=$(basename $BD); t0=$(date +%s)
  python3 /home/paperspace/logs/citrus_flip_block.py $BD >> $LOG 2>&1 || { say "$B FLIP REFUSED — skipped"; continue; }
  if grep -q "COLMAP" $BD/transforms.json; then say "$B already refined — skip"; continue; fi
  /home/paperspace/logs/citrus_refine_block.sh $BD >> $LOG 2>&1 || say "$B refine failed — odometry poses kept"
  say "$B: $(grep -oE 'p50 [0-9.]+ m p90 [0-9.]+ m|REFUSED[^;]*' $LOG | tail -1) ($(( $(date +%s)-t0 ))s)"
done
say "REFINE DONE: $(grep -l COLMAP $CFG/block_[0-9][0-9][0-9]/transforms.json | wc -l) blocks refined"
