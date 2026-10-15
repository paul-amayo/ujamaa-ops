#!/bin/bash
# e2_embedders.sh — train the Euclidean twin of every production embedder in the identity inventory (paper ablation arm B, 2026-10-13):
# same hierarchy, same recipe (c20cos20, 1500 ep, norms off, frozen curvature) + --euclidean; output <survey root>/experimental/identity/
# embedder/<survey>_euclid_v1. Sequential, skips finished ones.   e2_embedders.sh [inventory jsonl]
set -uo pipefail
INV=${1:-/home/paperspace/logs/identity_inventory.jsonl}; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/e2_embedders.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
python3 - "$INV" <<'PY' | while read -r SV ROOT HJ; do
import json, sys
seen = {}
for r in map(json.loads, open(sys.argv[1])):
    root = r['ws'].split('/experimental/')[0]; seen.setdefault((r['survey'], root, r['hj']), 1)
for (sv, root, hj) in seen: print(sv, root, hj)
PY
  OUT=$ROOT/experimental/identity/embedder; EN=${SV}_euclid_v1; [ -e $OUT/$EN/ckpts/model_best.pth ] && { say "$EN exists"; continue; }
  t0=$(date +%s); mkdir -p $OUT; say "=== $EN from $HJ"
  (cd $NS && nice -n 5 pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name $EN --output-dir $OUT \
     --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms --euclidean) > /home/paperspace/logs/e2_embedder_${EN}.log 2>&1
  say "$EN rc=$? in $(( $(date +%s)-t0 )) s: $(tr '\r' '\n' < /home/paperspace/logs/e2_embedder_${EN}.log | grep -a 'Row (retrieval)' | tail -1 | tr -s ' ')"
done
say "=== embedders done"
