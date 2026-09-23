#!/bin/bash
# Klapmuts-only H3DGS queue (09-23): the full queue is held until the disk has room for the citrus surveys.
L=/home/paperspace/logs/h3dgs_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
for S in /home/paperspace/data/klapmuts/apr_2026_zed /home/paperspace/data/klapmuts/dec_2025_ten_rows; do
  say "--- $(basename $S) (global-BA recipe; $(df -h / | awk 'NR==2{print $4}') free)"
  /home/paperspace/logs/h3dgs_survey.sh $S $S/experimental/h3dgs > /home/paperspace/logs/h3dgs_survey_$(basename $S).out 2>&1 && say "$(basename $S) OK" || say "$(basename $S) FAILED (see h3dgs_$(basename $S).log)"
done
say "KLAPMUTS QUEUE DONE"
