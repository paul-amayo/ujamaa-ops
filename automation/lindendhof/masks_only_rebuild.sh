#!/bin/bash
# masks_only_rebuild.sh (2026-10-17) — rebuild a LiDAR-anchored survey's identity reference as SAM3 masks only (MODE=masks_only in
# lindendhof_identity_build.py; Paul: "start with masks only") and redo the per-lane v3 identity on the same census, embedder and verdict
# frames. Nothing is deleted: the old reference -> supervision/trees_only_canopy_v1, the old per-lane identity dir -> identity_canopy_v1,
# the old text bank -> text_bank_canopy_v1.npz. Per lane: build maps, text bank, v3_identity_cell.sh (census seed, maps_v3, verdicts),
# argmax_eval.py (protocol_v3.jsonl for the paper's E1).
# Env: S, LANES ("1,2,..."), MASKS, EN; ROW_HALF, TREE_HALF, WSMAP, SITE as for anchor_identity_chain.sh
set -uo pipefail
: ${S:?} ${LANES:?} ${MASKS:?} ${EN:?}
ID=$S/experimental/identity; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native
LD=/home/paperspace/code/automation/lindendhof; EMB=$ID/embedder/$EN/ckpts/model_best.pth
L=/home/paperspace/logs/masks_only_$(basename $S).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s)
say "=== masks-only reference: $S lanes $LANES masks $MASKS widths ${ROW_HALF:-0.6}/${TREE_HALF:-0.6} embedder $EN"
[ -e $EMB ] || { say "embedder missing: $EMB"; exit 1; }
for n in $(echo $LANES | tr ',' ' '); do C=$ID/cells/lane$n
  [ -e $C/supervision/trees_only_canopy_v1 ] || mv $C/supervision/trees_only $C/supervision/trees_only_canopy_v1
  [ -e $C/text_bank_canopy_v1.npz ] || { [ -e $C/text_bank.npz ] && mv $C/text_bank.npz $C/text_bank_canopy_v1.npz; }
done
(cd $NS && MODE=masks_only pixi run python $LD/lindendhof_identity_build.py $S $ID $LANES) 2>&1 | grep -a '^\[build\]\|Error\|Traceback' | tee -a $L
for n in $(echo $LANES | tr ',' ' '); do C=$ID/cells/lane$n; WSN=$(python3 -c "import os; m=dict(kv.split(':') for kv in os.environ.get('WSMAP','').split(',') if kv); print(m.get('$n', 'chunk_lane$n'))"); WS=$S/experimental/v3/$WSN
  [ -e $C/supervision/trees_only/manifest.json ] || { say "lane $n: no reference built"; continue; }
  [ -e $WS/identity_canopy_v1 ] || mv $WS/identity $WS/identity_canopy_v1
  (cd $NS && pixi run python $NT/build_text_bank.py --manifest $C/supervision/trees_only/manifest.json --hierarchy-json $ID/marker_hierarchy.json --out $C/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
  CFG=$WS/outputs/sf_60000_bilateral/splatfacto/run/config.yml CELL=$C EMB=$EMB HJ=$ID/marker_hierarchy.json KF=$WS/images OUT=$WS/identity SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh > /dev/null 2>&1
  say "lane $n: $(grep -a '\[v3 means\]' /home/paperspace/logs/v3_identity_$(basename $S)_$WSN.log 2>/dev/null | tail -1 | cut -c1-300)"
  (cd $NS && pixi run python $V3/argmax_eval.py --identity $WS/identity --tag v3) 2>&1 | grep -av 'Warning\|warn' | tail -2 | tee -a $L
done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
