#!/bin/bash
# Combined H3DGS queue (09-23, after the Klapmuts fixes): ten_rows first (Paul's order), then citrus 01-04, then 05 v2.
# Waits for the apr_2026_zed finish+serve chain so two survey pipelines never share the GPU.
L=/home/paperspace/logs/h3dgs_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until grep -q "SERVE DONE" /home/paperspace/logs/tassili_serve.log 2>/dev/null && ! pgrep -f "^/bin/bash /home/paperspace/logs/h3dgs_survey.sh /home/paperspace/data/klapmuts/apr" > /dev/null; do sleep 60; done
for S in /home/paperspace/data/klapmuts/dec_2025_ten_rows /home/paperspace/data/citrus_all/01_13B_Jackal /home/paperspace/data/citrus_all/02_13B_Jackal /home/paperspace/data/citrus_all/03_13B_Jackal /home/paperspace/data/citrus_all/04_13D_Jackal; do
  say "--- $(basename $S) ($(df -h / | awk 'NR==2{print $4}') free)"
  /home/paperspace/logs/h3dgs_survey.sh $S $S/experimental/h3dgs > /home/paperspace/logs/h3dgs_survey_$(basename $S).out 2>&1 && say "$(basename $S) OK" || say "$(basename $S) FAILED (see h3dgs_$(basename $S).log)"
done
say "--- 05_13D_Jackal v2 (global-BA recipe)"
/home/paperspace/logs/h3dgs_survey.sh /home/paperspace/data/citrus_all/05_13D_Jackal /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_v2 > /home/paperspace/logs/h3dgs_survey_05_v2.out 2>&1 && say "05 v2 OK" || say "05 v2 FAILED"
say "ALL QUEUE DONE"
