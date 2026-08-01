#!/bin/bash
# S4c (TRUE strict-only, v2_S2 clean base) — waits for the weekend queue.
until grep -q "WEEKEND-ALL-DONE" /home/paperspace/logs/weekend_queue.log 2>/dev/null; do sleep 600; done
while ps -eo cmd | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 60; done
exec /home/paperspace/code/automation/s4c_clean.sh
