#!/bin/bash
# e2_arms_run.sh — the geometry ablation (e2_arms_cell.sh: arms B, C, D) over every cell of the identity inventory (2026-10-13).
# Arm B needs the survey's Euclidean twin embedder (e2_embedders.sh); cells whose twin is missing run C and D only.
#   e2_arms_run.sh [inventory jsonl]   env: ARMS=B,C,D
set -uo pipefail
INV=${1:-/home/paperspace/logs/identity_inventory.jsonl}; ARMS=${ARMS:-B,C,D}; V3=/home/paperspace/code/automation/h3dgs/v3
L=/home/paperspace/logs/e2_arms.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); n=0
say "=== E2 arms $ARMS over $(wc -l < $INV) cells"
while read -r line; do
  eval $(python3 -c "import json,sys; r=json.loads(sys.argv[1]); print(f\"N={r['identity']} SV={r['survey']} ROOT={r['ws'].split('/experimental/')[0]}\")" "$line")
  EMB=$ROOT/experimental/identity/embedder/${SV}_euclid_v1/ckpts/model_best.pth; A=$ARMS
  [ -e $EMB ] || { A=$(echo $ARMS | sed 's/B,//; s/,B//; s/^B$//'); [ -z "$A" ] && continue; }
  done_all=1; for arm in $(echo $A | tr ',' ' '); do case $arm in B) f=euclid;; C) f=flat96;; C30) f=flat30;; D) f=hyper_own;; *) f=$arm;; esac; [ -s $N/protocol_$f.jsonl ] || done_all=0; done
  [ $done_all = 1 ] && continue
  N=$N EMB_EUCLID=$EMB ARMS=$A bash $V3/e2_arms_cell.sh > /dev/null 2>&1; n=$((n+1))
done < $INV
say "=== done: $n cells in $(( ($(date +%s)-t0)/60 )) min"
