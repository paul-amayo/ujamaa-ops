#!/bin/bash
# wait_for.sh <done file> <pid> <timeout s>: returns when the DONE file exists, or the PID is gone, or the timeout passes (A100_RULES item 3)
F=$1; PID=$2; T=$3; t0=$(date +%s)
while true; do
  [ -f "$F" ] && { echo "DONE file: $(cat $F)"; exit 0; }
  kill -0 $PID 2>/dev/null || { sleep 5; [ -f "$F" ] && { echo "DONE file: $(cat $F)"; exit 0; }; echo "PID $PID gone without DONE file"; exit 2; }
  [ $(( $(date +%s) - t0 )) -gt $T ] && { echo "timeout after $T s"; exit 3; }
  sleep 15
done
