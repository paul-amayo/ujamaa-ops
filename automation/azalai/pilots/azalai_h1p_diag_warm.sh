#!/bin/bash
# azalai_h1p_diag_warm.sh (11 Nov 2026, A100 session) - POST HOC DIAGNOSTIC, not part of the design; no rule depends on it.
# The control (frozen wording, first request after a fresh model load) differed from v1 seed 0 in the twin arm only. Hypothesis:
# llama-server reuses the cached prompt prefix of the previous request, so the twin system prompt's KV is computed from a different
# offset when the previous request was an enumerated one (v1 seed 0: after the dry run's enumerated arm; sets A and B: after an
# enumerated run) than on a cold start (the control). Test: the frozen bank, twin arm only, run right after the chain's last
# request (enumerated), with the v1 harness unchanged. Prediction under the hypothesis: identical to v1 seed 0, reply for reply.
W=/mnt/shared/azalai_h1p; PY="/home/paperspace/.pixi/bin/pixi run python"; cd /home/paperspace/code/nerf_new || exit 1
[ -e $W/DIAG_STARTED ] && { echo "diagnostic already started: $(cat $W/DIAG_STARTED)"; exit 1; }
ME=$$; echo "$ME $(date -u +%FT%TZ) ollama ps: $(curl -s -m 5 localhost:11434/api/ps | head -c 300)" > $W/DIAG_STARTED
( i=0; while [ ! -e $W/DIAG_DONE ] && [ ! -e $W/DIAG_FAILED ] && kill -0 $ME 2>/dev/null && [ $i -lt 1800 ]; do
    echo "$(date -u +%T) n=$(ss -tnH state established '( dport = :11434 )' 2>/dev/null | wc -l) $(ss -tnpH state established '( dport = :11434 )' 2>/dev/null | grep -o 'pid=[0-9]*' | sort -u | tr '\n' ' ')"
    i=$((i + 1)); sleep 2; done; echo "$(date -u +%T) watcher end" ) > $W/h1p_diag_warm_clients.log 2>&1 &
$PY $W/bin/azalai_h1_ground_v1.py $W $W/h1p_diag_warm --seed 0 --arms twin > $W/h1p_diag_warm.log 2>&1 && touch $W/DIAG_DONE || touch $W/DIAG_FAILED
echo "$(date -u +%FT%TZ) end" >> $W/DIAG_STARTED
