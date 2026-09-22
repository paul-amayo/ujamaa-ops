#!/bin/bash
# H3DGS survey queue (Paul, 09-22): after the 05 run, Klapmuts (apr_2026_zed, dec_2025_ten_rows), then citrus 01-04.
# Each survey runs h3dgs_survey.sh (idempotent); a failed survey is logged and the queue moves on.
L=/home/paperspace/logs/h3dgs_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
while pgrep -f "h3dgs_resume.sh" > /dev/null; do sleep 120; done
say "05 runner finished — queue starts"
for S in /home/paperspace/data/klapmuts/apr_2026_zed /home/paperspace/data/klapmuts/dec_2025_ten_rows \
         /home/paperspace/data/citrus_all/01_13B_Jackal /home/paperspace/data/citrus_all/02_13B_Jackal \
         /home/paperspace/data/citrus_all/03_13B_Jackal /home/paperspace/data/citrus_all/04_13D_Jackal; do
  say "--- $(basename $S)"
  /home/paperspace/logs/h3dgs_survey.sh $S $S/experimental/h3dgs > /home/paperspace/logs/h3dgs_survey_$(basename $S).out 2>&1 && say "$(basename $S) OK" || say "$(basename $S) FAILED (see h3dgs_$(basename $S).log)"
done
say "QUEUE DONE"
