#!/bin/bash
# Queue the fruit densification behind the 05 chunk side-car fleet so it does not contend with the 04 H3DGS pipeline
# for the last of the GPU (33.9 of 41 GB in use when this was written). Waits for the chunk loop to exit, then runs
# block 021 first (side-car seed-only 0.002 vs block-style 0.578 — the clearest A/B), then 018 and 019.
set -uo pipefail
L=/home/paperspace/logs/sidecar_05_13D_Jackal_fruitdensify.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "queued: waiting for the 05 chunk side-car fleet to finish"
while pgrep -f 'sidecar_chunk\.sh 05_13D_Jackal' > /dev/null 2>&1; do sleep 60; done
say "chunk fleet done; GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used"
for N in 021 018 019; do
  bash /home/paperspace/logs/sidecar_fruit_densify.sh 05_13D_Jackal $N 2000 0.1 || say "block $N FAILED, continuing"
done
say "fruit densification sweep done"
