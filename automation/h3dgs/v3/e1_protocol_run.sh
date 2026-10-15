#!/bin/bash
# e1_protocol_run.sh — the paper's E1 protocol scorer (argmax_eval.py) over every cell of an identity inventory (2026-10-13).
# Sequential, GPU-light (one decode per frame), nice'd beside a training fleet; skips cells that already have protocol_<TAG>.jsonl.
#   e1_protocol_run.sh [inventory jsonl = /home/paperspace/logs/identity_inventory.jsonl]   env: TAG=v3 CHECK=1
set -uo pipefail
INV=${1:-/home/paperspace/logs/identity_inventory.jsonl}; TAG=${TAG:-v3}; CHECK=${CHECK:-1}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/e1_protocol.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== E1 protocol ($TAG) over $(wc -l < $INV) cells ($INV)"; t0=$(date +%s); n=0
while read -r line; do
  N=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['identity'])" "$line"); S=$(python3 -c "import json,sys; r=json.loads(sys.argv[1]); print(r['survey']+'/'+r['chunk'])" "$line")
  [ -s $N/protocol_$TAG.jsonl ] && continue
  (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $N --tag $TAG $([ "$CHECK" = 1 ] && echo --check-oracle) 2>&1) | grep -aE '^\[(protocol|check|vocab)\]|Error|Traceback' | sed "s#^#[$S] #" | tee -a $L
  n=$((n+1))
done < $INV
say "=== done: $n cells scored in $(( ($(date +%s)-t0)/60 )) min"
