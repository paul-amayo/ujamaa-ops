#!/bin/bash
# demo_sheets_waiter.sh — after the whole overnight chain (fruit v2 -> 01 retry [+ 05 fruit re-composite] -> 05 0_0 cut),
# (re)build contact_sheet.png beside every reel so the dashboard can fetch mp4 + sheet together. CPU only.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until grep -qE 'demo_chunks_00 done|05_0_0: backdrop FAILED' $L 2>/dev/null; do sleep 120; done
for d in /home/paperspace/logs/demo_chunks/*/; do
  [ -e $d/demo.mp4 ] && [ -e $d/demo_script.json ] && /home/paperspace/envs/match/bin/python /home/paperspace/code/automation/h3dgs/demo_contact_sheet.py $d 2>&1 | grep -aE '^\[sheet\]|Error|Traceback' | tee -a $L
done
say "=== contact sheets done: $(ls /home/paperspace/logs/demo_chunks/*/contact_sheet.png 2>/dev/null | xargs -n1 dirname | xargs -n1 basename | paste -sd' ')"
