#!/bin/bash
# sidecar_full.sh <survey id> <NNN> [margin m=8] [min opacity=0.95] — the whole containment side-car recipe for one block:
#   1. sidecar_block_glref.sh : H3DGS leaves -> stage-1 ckpt -> census-init seed (recipe of record) -> plain verdict
#   2. sidecar_bg_reseed.sh   : census WITH the unlabelled row, seed with --bg-competes (void gaussians carry no identity)
#   3. sidecar_boost_score.sh : identity-carrying gaussians raised to >= min opacity (nearest labelled surface owns the pixel)
# and a one-line summary: plain / bg / bg+boost IoUs beside the block-style glref verdict.
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; MARGIN=${3:-8}; MINOP=${4:-0.95}
L=/home/paperspace/logs/sidecar_${SV}.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
cd /home/paperspace/logs; t0=$(date +%s)
bash /home/paperspace/logs/sidecar_block_glref.sh $SV $NNN $MARGIN > /home/paperspace/logs/sidecar_${SV}_b${NNN}_full1.out 2>&1 || { say "block $NNN: step 1 failed"; exit 1; }
bash /home/paperspace/logs/sidecar_bg_reseed.sh $SV $NNN sidecar 1.0 > /home/paperspace/logs/sidecar_${SV}_b${NNN}_full2.out 2>&1 || { say "block $NNN: step 2 failed"; exit 1; }
bash /home/paperspace/logs/sidecar_boost_score.sh $SV $NNN sidecar glref_bg_f1.0 $MINOP > /home/paperspace/logs/sidecar_${SV}_b${NNN}_full3.out 2>&1 || { say "block $NNN: step 3 failed"; exit 1; }
iou(){ grep -aE "^\[$NNN $1" $VL | grep -oE '(TREE|ROW|FRUIT) [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g; s/(TREE|ROW|FRUIT) [0-9]+ "([a-z]+)": thr [0-9.]+ IoU ([0-9.]+)/\1 \2 \3/' | tr '\n' ' '; }
say "SUMMARY block $NNN ($(( ($(date +%s)-t0)/60 )) min): plain [$(iou "kf_")] | bg [$(iou "sidecar bg_f1.0")] | bg+boost [$(iou "sidecar glref_bg_f1.0_op${MINOP}")] | BLOCK-STYLE $(python3 -c "
import json; d=dict(json.load(open('/home/paperspace/data/citrus_all/$SV/prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_glref.json'))['blocks']).get('$NNN'); print(' '.join(f'{k} {v}' for k, v in list(d['trees'].items()) + list(d['rows'].items())) if d else 'none')")"
