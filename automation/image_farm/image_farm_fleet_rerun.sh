#!/bin/bash
# After the running fleet completes, run it once more: only segments without a splat.ply train (retries the segments
# that failed a gate before the gate became advisory, e.g. IMG_7964_s0 clustering).
while pgrep -f "^bash /home/paperspace/logs/image_farm_fleet.sh" > /dev/null; do sleep 120; done
echo "[$(date '+%m-%d %H:%M:%S')] RERUN first fleet pass finished; second pass for missing segments"
exec bash /home/paperspace/logs/image_farm_fleet.sh
