#!/bin/bash
# azalai_h1p_watch2.sh (11 Nov 2026, A100 session) - supplementary watcher, started during the run (20:14 UTC): every 2 s the
# NUMBER of established client connections to the ollama port (all users), beside the pids the chain's watcher logs (a client of
# another user would show no pid there). Ends when the chain pid is gone, on DONE / FAILED, or after 3 h.
W=/mnt/shared/azalai_h1p; CH=$1; i=0
while kill -0 $CH 2>/dev/null && [ ! -e $W/DONE ] && [ ! -e $W/FAILED ] && [ $i -lt 5400 ]; do
  n=$(ss -tnH state established '( dport = :11434 )' 2>/dev/null | wc -l)
  p=$(ss -tnpH state established '( dport = :11434 )' 2>/dev/null | grep -o 'pid=[0-9]*' | sort -u | tr '\n' ' ')
  echo "$(date -u +%T) n=$n $p"; i=$((i + 1)); sleep 2
done; echo "$(date -u +%T) watch2 end"
