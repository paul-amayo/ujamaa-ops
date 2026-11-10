#!/bin/bash
# run_lerf_queue.sh <wait_file> <survey:chunk> ... — ovbench (2026-11-10): after <wait_file> exists (05 1_0 end to end), run
# run_lerf_cell.sh on each cell in turn; before each, wait until the card has >= 16 GB free (shared A100).
#   setsid nohup nice -n 10 bash run_lerf_queue.sh /home/paperspace/logs/ovbench/lerf/05_13D_Jackal_1_0/ALL_DONE 01_13B_Jackal:chunk_0_2 ... &
WAIT=$1; shift; B=/home/paperspace/logs/ovbench
until [ -f $WAIT ]; do sleep 60; done
for sc in "$@"; do
  S=${sc%%:*}; C=${sc##*:}
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 16000 ]; do sleep 60; done
  echo "[queue] $S $C start $(date -Is)"
  bash $B/bin/run_lerf_cell.sh $S $C > $B/lerf/${S}_${C#chunk_}.chain.log 2>&1
  echo "[queue] $S $C rc=$? $(date -Is)"
done
echo "[queue] done $(date -Is)"
