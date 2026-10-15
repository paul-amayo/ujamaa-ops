#!/bin/bash
# e2_rescore_rv.sh — re-score the Euclidean arm (B) with the RELATIVE presence check (2026-10-13): the April/Dec/Lindendhof Euclidean twins seed
# at norm ~1.0 (citrus twins ~5, like the hyperbolic tangents), so containment_eval's absolute ||f|| < 0.5 voided most of their blended pixels
# (April trees 0.000). argmax_eval.py --void-ref scales the check to the arm's own seed norms. The old file is kept as protocol_euclid_abs05.jsonl.
#   e2_rescore_rv.sh [inventory jsonl] [survey regex = non-citrus]
set -uo pipefail; INV=${1:-/home/paperspace/logs/identity_inventory_v2.jsonl}; RE=${2:-apr_2026_zed|dec_2025_ten_rows|nov_2025_a300|IMG_}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/e2_rescore_rv.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; say "=== relative-void re-score of arm B for /$RE/"
grep -E "\"survey\": \"($RE)" $INV | while read -r line; do
  N=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['identity'])" "$line"); S=$(echo $N | awk -F/ '{print $6"/"$8}')
  [ -e $N/arms/features_euclid.bin ] || { say "$S: no euclid seed"; continue; }; [ -e $N/protocol_euclid_abs05.jsonl ] && continue
  [ -e $N/protocol_euclid.jsonl ] && mv $N/protocol_euclid.jsonl $N/protocol_euclid_abs05.jsonl
  (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $N --tag euclid --space euclid --emb $(python3 -c "import json; print(json.load(open('$N/arms/features_euclid.bin.json'))['embedder'])") --maps arms/maps_euclid/%s.npz --void-ref $N/arms/features_euclid.bin $N/features_B.bin 2>&1) | grep -aE '^\[(void|protocol)\]|Error|Traceback' | sed "s#^#[$S] #" | tee -a $L
done; say "=== done"
