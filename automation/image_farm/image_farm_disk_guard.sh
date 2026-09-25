#!/bin/bash
# Disk watchdog for the image_farm fleet: when free space on /home/paperspace/data drops below MIN_GB, stop the fleet,
# the prep sweep and the second-pass runner (image_farm_stop.sh, anchored patterns) and log it. Never deletes anything.
MIN_GB=${IF_MIN_FREE_GB:-45}
while true; do
  free=$(df --output=avail -BG /home/paperspace/data | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -lt "$MIN_GB" ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] DISK GUARD: ${free} GB free < ${MIN_GB} GB — stopping the fleet, prep sweep and rerun runner"
    pkill -f "^bash /home/paperspace/logs/image_farm_fleet_rerun.sh"; pkill -f "^bash /home/paperspace/logs/image_farm_prep_all.sh"
    bash /home/paperspace/logs/image_farm_stop.sh
    echo "[$(date '+%m-%d %H:%M:%S')] DISK GUARD: stopped; nothing deleted — Paul decides what to clear"
    exit 0
  fi
  sleep 60
done
