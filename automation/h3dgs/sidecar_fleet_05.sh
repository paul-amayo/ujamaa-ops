#!/bin/bash
# sidecar_fleet_05.sh — ratio-2 containment side-cars for every 05 block (Paul, 2026-09-27: the ask-the-orchard / lift-off video
# needs identity on the whole survey). Sequential; skips blocks whose seed exists; stops if the data volume drops under 25 G free.
L=/home/paperspace/logs/sidecar_05_13D_Jackal.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== fleet start"
for b in $(seq -f %03g 0 42); do
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free" -ge 25 ] || { say "fleet STOPPED at block $b: ${free}G free"; exit 1; }
  bash /home/paperspace/logs/sidecar_r2.sh 05_13D_Jackal $b 8 > /home/paperspace/logs/sidecar_05_b${b}_r2.out 2>&1
done
say "=== fleet DONE"
