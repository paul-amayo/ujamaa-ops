#!/bin/bash
# dec_2025_ten_rows: re-ingest the A300 mcap -> monolithics on the A100.
# Canonical extractor (a300_bag_to_monolithics.py), run under nerf_new's
# python3.10 because that is where the aru_py_logger writer binding (cp310)
# and rosbags live on this box. Validated V100 run: 32.2 min, every stream
# count == the bag's connection table (image 5903, laser 3943, zed_odom 5914).
set -uo pipefail
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows
M=$R/prod/monos
OUT=$M/monolithics
BAG=$M/rosbag2_2025_12_03-13_22_24_0.mcap
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10
SRC=/home/paperspace/code/aru_sil_core/src/scripts
LOG=$R/_logs/ten_rows_ingest_a100.log
say(){ echo "[$(date '+%H:%M:%S')] INGEST $*"; }
export PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib:/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib
say "start: $(basename $BAG) -> $OUT"
t0=$(date +%s)
QT_QPA_PLATFORM=offscreen env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY $SRC/a300_bag_to_monolithics.py \
  --bag $BAG --out $OUT --rig $M/rig.json > $LOG 2>&1 || { say "INGEST FAILED (see $LOG)"; exit 1; }
say "done in $(( ($(date +%s)-t0)/60 )) min"
grep -E "DONE|extracting|absent" $LOG | tail -3
echo "--- streams written ---"; ls -la $OUT | grep -E "\.monolithic$|npz"
say "INGEST DONE"
