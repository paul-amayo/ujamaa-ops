#!/bin/bash
# v3_phone_prep.sh — single-chunk projects (image_farm_h3dgs_prep.py: undistorted PINHOLE images, poses from transforms.json, init.ply,
# every 10th frame held out) for every image-farm base segment and GenDIA survey without one, then a v3 fleet queue for them
# (2026-10-12, Paul: identity for "all the image farm runs and gendia" — a v3 RGB model first). Variants (_cabbage, _sall, _s1a,
# _walk_demo) and segments without transforms.json are skipped; segments that already carry a v3 lane model are left out.
Q=/home/paperspace/logs/v3_fleet_queue_phone_20261012.txt; L=/home/paperspace/logs/v3_phone_prep.log; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
P=/home/paperspace/code/automation/image_farm/image_farm_h3dgs_prep.py; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
echo "# v3 phone-survey queue (2026-10-12): <segment root> <project> lane 0  (RGB first; identity per survey family afterwards)" > $Q
for S in /home/paperspace/data/gendia/{osuga,osuga_baseline,sukuma,mango,pawpaw,osuga_full} $(ls -d /home/paperspace/data/image_farm/*/*/IMG_*_s[0-9]); do
  n=$(basename $S); [ -e $S/transforms.json ] || { say "$n: no transforms.json - skipped"; continue; }
  [ -e $S/experimental/v3/chunk_lane/outputs/sf_60000_bilateral/eval_heldout.json ] && { say "$n: v3 model exists - skipped"; continue; }
  [ -e $S/h3dgs/camera_calibration/chunks/lane/sparse/0/images.bin ] || $PY $P $S --proj h3dgs 2>&1 | tail -1 | tee -a $L
  [ -e $S/h3dgs/camera_calibration/chunks/lane/sparse/0/images.bin ] && echo "$S $S/h3dgs lane 0" >> $Q || say "$n: prep FAILED"
done
say "=== $(grep -vc '^#' $Q) entries queued in $Q"
