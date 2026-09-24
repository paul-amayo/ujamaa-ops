#!/bin/bash
# Wait for the foreground validation prep (IMG_7999) to exit, then start the sequential image_farm fleet.
# Anchored pgrep pattern (the python interpreter path) so this script's own command line can never match.
while pgrep -f "^/home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/logs/image_farm_prep.py" > /dev/null; do sleep 20; done
echo "[$(date '+%m-%d %H:%M:%S')] LAUNCH validation prep finished; starting fleet"
exec bash /home/paperspace/logs/image_farm_fleet.sh
