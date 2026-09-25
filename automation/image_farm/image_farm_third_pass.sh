#!/bin/bash
# After the second fleet pass exits: one more pass restricted to the clip whose segment needed the name repair.
while pgrep -f "^bash /home/paperspace/logs/image_farm_fleet.sh" > /dev/null || pgrep -f "^bash /home/paperspace/logs/image_farm_fleet_rerun.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%m-%d %H:%M:%S')] THIRD PASS: IMG_7961 (s1 repaired names)"
IF_CLIPS="IMG_7961" exec bash /home/paperspace/logs/image_farm_fleet.sh
