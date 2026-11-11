#!/bin/bash
# azalai_h1p_run_a100.sh (11 Nov 2026, A100 session) - the AZ-3 run chain of azalai_h1p_run.sh (Mac agent, ops d1b490e), versioned
# for the box. Same runs, same order, same harness call (azalai_h1_ground_v1.py unchanged, --seed 0 --bank <file>), twin then
# enumerated inside each run, one process at a time. Differences from azalai_h1p_run.sh, all operational:
#  (1) before each run, if the model is not loaded in ollama, wait for >= 16 GB free GPU memory (the A100 session's rule for
#      this test; the design's 12 GB is the floor), bounded at 120 min per run, then FAILED;
#  (2) a run that exits non-zero is retried at most twice, 5 min apart (the harness resumes: an instruction already in the
#      record file is skipped); every attempt is logged in chain.log and appended to <run>.log;
#  (3) the watcher also stops when this chain's pid is gone (and after 6 h), and logs `ollama ps` every 30 s, so that a model
#      loaded with part of it on the CPU (size_vram < size) would be seen.
# usage (box): setsid nohup bash /mnt/shared/azalai_h1p/bin/azalai_h1p_run_a100.sh > /mnt/shared/azalai_h1p/chain.log 2>&1 < /dev/null &
W=/mnt/shared/azalai_h1p; PY="/home/paperspace/.pixi/bin/pixi run python"; cd /home/paperspace/code/nerf_new || exit 1
[ -e $W/STARTED ] && { echo "chain already started: $(cat $W/STARTED)"; exit 1; }      # v1's incident: never two chains
CH=$$; echo "$CH $(date -u +%FT%TZ) azalai_h1p_run_a100.sh" > $W/STARTED
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
# watcher: every 2 s, the client processes connected to the ollama port (a foreign pid during a run is a concurrent request)
( i=0
  while [ ! -e $W/DONE ] && [ ! -e $W/FAILED ] && kill -0 $CH 2>/dev/null && [ $i -lt 10800 ]; do
    echo "$(date -u +%T) $(ss -tnpH state established '( dport = :11434 )' 2>/dev/null | grep -o 'pid=[0-9]*' | sort -u | tr '\n' ' ')"
    [ $((i % 15)) -eq 0 ] && echo "$(date -u +%T) PS $(curl -s -m 5 localhost:11434/api/ps | head -c 600)"
    i=$((i + 1)); sleep 2
  done; echo "$(date -u +%T) watcher end" ) > $W/ollama_clients.log 2>&1 &
echo $! > $W/watcher.pid
run() {   # run <name> <bank file>
  local t0 free loaded attempt
  t0=$(date +%s)
  for attempt in 1 2 3; do
    while :; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      loaded=$(curl -s -m 5 localhost:11434/api/ps | grep -c gemma-4-12b)
      { [ "${loaded:-0}" -ge 1 ] || [ "${free:-0}" -ge 16000 ]; } && break
      if [ $(( $(date +%s) - t0 )) -gt 7200 ]; then echo "[chain] $(date -u +%T) $1: GPU short for 120 min, stopping"; touch $W/FAILED; exit 1; fi
      echo "[chain] $(date -u +%T) GPU free ${free} MiB < 16000 and model not loaded: waiting"; sleep 60
    done
    echo "[chain] $(date -u +%FT%TZ) start $1 attempt $attempt (GPU free ${free} MiB; ollama ps: $(curl -s -m 5 localhost:11434/api/ps | head -c 300))"
    if $PY $W/bin/azalai_h1_ground_v1.py $W $W/$1 --seed 0 --bank $W/$2 >> $W/$1.log 2>&1; then
      echo "[chain] $(date -u +%FT%TZ) end $1: $(wc -l < $W/$1_twin.jsonl) twin, $(wc -l < $W/$1_enumerated.jsonl) enumerated records"
      return 0
    fi
    echo "[chain] $(date -u +%FT%TZ) $1 attempt $attempt exited non-zero"
    [ $attempt -lt 3 ] && sleep 300
  done
  echo "[chain] $(date -u +%FT%TZ) $1 FAILED after 3 attempts"; touch $W/FAILED; exit 1
}
run h1p_control instruction_bank_v0.csv
run h1p_setA h1p_bank_setA.csv
run h1p_setB h1p_bank_setB.csv
run h1p_setA_change_rerun h1p_bank_setA_change.csv
run h1p_setB_change_rerun h1p_bank_setB_change.csv
run h1p_corA h1p_bank_corA.csv
run h1p_corB h1p_bank_corB.csv
echo "[chain] $(date -u +%FT%TZ) all runs done"; touch $W/DONE
