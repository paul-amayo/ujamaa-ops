#!/bin/bash
# azalai_h1p_run.sh (11 Nov 2026) — H1 paraphrase test (AZ-3): the run chain on the box, one process at a time, in the order of
# the design of record (H1_paraphrase_design.md Section 3). The v1 harness is called unchanged with --bank.
# usage (box): setsid nohup bash /mnt/shared/azalai_h1p/bin/azalai_h1p_run.sh > /mnt/shared/azalai_h1p/chain.log 2>&1 &
W=/mnt/shared/azalai_h1p; PY="/home/paperspace/.pixi/bin/pixi run python"; cd /home/paperspace/code/nerf_new || exit 1
[ -e $W/STARTED ] && { echo "chain already started: $(cat $W/STARTED)"; exit 1; }      # v1's incident: never two chains
echo "$$ $(date -u +%FT%TZ)" > $W/STARTED
# watcher: every 2 s, the client processes connected to the ollama port (a foreign pid during a run would be a concurrent request)
( while [ ! -e $W/DONE ] && [ ! -e $W/FAILED ]; do echo "$(date -u +%T) $(ss -tnpH state established '( dport = :11434 )' 2>/dev/null | grep -o 'pid=[0-9]*' | sort -u | tr '\n' ' ')"; sleep 2; done ) > $W/ollama_clients.log 2>&1 &
echo $! > $W/watcher.pid
run() {   # run <name> <bank file>
  while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); loaded=$(curl -s localhost:11434/api/ps | grep -c gemma-4-12b)
    [ "$loaded" -ge 1 ] || [ "$free" -ge 12000 ] && break; echo "[chain] $(date -u +%T) GPU free ${free} MiB < 12000 and model not loaded: waiting"; sleep 60; done
  echo "[chain] $(date -u +%FT%TZ) start $1 (GPU free ${free} MiB; ollama ps: $(curl -s localhost:11434/api/ps | head -c 300))"
  $PY $W/bin/azalai_h1_ground_v1.py $W $W/$1 --seed 0 --bank $W/$2 > $W/$1.log 2>&1 || { echo "[chain] $1 FAILED"; touch $W/FAILED; exit 1; }
  echo "[chain] $(date -u +%FT%TZ) end $1: $(wc -l < $W/$1_twin.jsonl) twin, $(wc -l < $W/$1_enumerated.jsonl) enumerated records"
}
run h1p_control instruction_bank_v0.csv
run h1p_setA h1p_bank_setA.csv
run h1p_setB h1p_bank_setB.csv
run h1p_setA_change_rerun h1p_bank_setA_change.csv
run h1p_setB_change_rerun h1p_bank_setB_change.csv
run h1p_corA h1p_bank_corA.csv
run h1p_corB h1p_bank_corB.csv
echo "[chain] $(date -u +%FT%TZ) all runs done"; touch $W/DONE
