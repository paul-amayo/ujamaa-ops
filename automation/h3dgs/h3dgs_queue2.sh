#!/bin/bash
# After the main queue: re-run 05_13D with the global-BA recipe in a fresh project dir (the first 05 run used
# per-chunk BA and is kept for the seam/PSNR record and the border-misregistration measurement).
L=/home/paperspace/logs/h3dgs_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until grep -q "QUEUE DONE" $L 2>/dev/null; do sleep 300; done
say "--- 05_13D_Jackal v2 (global-BA recipe)"
/home/paperspace/logs/h3dgs_survey.sh /home/paperspace/data/citrus_all/05_13D_Jackal /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_v2 > /home/paperspace/logs/h3dgs_survey_05_v2.out 2>&1 && say "05 v2 OK" || say "05 v2 FAILED (see h3dgs_05_13D_Jackal.log)"
say "QUEUE2 DONE"
