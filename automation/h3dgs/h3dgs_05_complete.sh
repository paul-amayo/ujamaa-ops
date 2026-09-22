#!/bin/bash
# After the first 05 runner exits, re-run the idempotent runner once so skipped steps (e.g. chunk 1_0's
# post-opt, which OOM'd next to the hierarchy backend) are completed before the queue moves to Klapmuts.
while pgrep -f "h3dgs_resume.sh" > /dev/null; do sleep 5; done
echo "[$(date '+%H:%M:%S')] first 05 runner finished — completion pass" | tee -a /home/paperspace/logs/h3dgs_train.log
/home/paperspace/logs/h3dgs_resume.sh
