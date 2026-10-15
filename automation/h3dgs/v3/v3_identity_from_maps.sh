#!/bin/bash
# v3_identity_from_maps.sh — v3 identity for a trained v3 workspace whose survey already has per-keyframe supervision maps
# (2026-10-12): v3_cell_from_maps.py (links + merged manifest + split) -> text bank -> verdict frames (pick_frames.py) ->
# v3_identity_cell.sh (flat census, seed B, feature maps, containment verdicts). Env: WS (v3 workspace), SUPGLOB (trees_only dirs),
# EMB, HJ, KF (images dir for verdict figures), [KFMAP] (lanes), [RUN=sf_60000_bilateral], [MUST] (verdict frames to include).
set -uo pipefail
: ${WS:?} ${SUPGLOB:?} ${EMB:?} ${HJ:?} ${KF:?}
RUN=${RUN:-sf_60000_bilateral}; V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native; NS=/home/paperspace/code/nerf_new
CELL=$WS/cell; L=/home/paperspace/logs/v3_idmaps_$(echo $WS | sed 's#/home/paperspace/data/##; s#/experimental/v3/chunk_#_#; s#/#_#g').log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
t0=$(date +%s); say "=== v3 identity from existing maps: $WS | maps $SUPGLOB | emb $EMB | hier $HJ ${KFMAP:+| kf_map $KFMAP}"
python3 $V3/v3_cell_from_maps.py --ws $WS --sup-glob "$SUPGLOB" --out $CELL ${KFMAP:+--kf-map $KFMAP} 2>&1 | tee -a $L
[ -e $CELL/supervision/trees_only/manifest.json ] || { say "cell FAILED"; exit 1; }
[ -e $CELL/text_bank.npz ] || (cd $NS && pixi run python $NT/build_text_bank.py --manifest $CELL/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $CELL/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
[ -e $CELL/frames.json ] || python3 $NT/pick_frames.py --names $CELL/split_names.json --sup $CELL/supervision/trees_only --must ${MUST:-} --n-train 4 --n-eval 10 --out $CELL/frames.json | tee -a $L
CFG=$WS/outputs/$RUN/splatfacto/run/config.yml CELL=$CELL EMB=$EMB HJ=$HJ KF=$KF OUT=$WS/identity SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh > /dev/null 2>&1
grep -a '\[v3 means\]' /home/paperspace/logs/v3_identity_$(echo $WS/identity | cut -d/ -f6)_$(basename $WS).log 2>/dev/null | tail -1 | tee -a $L
say "done in $(( ($(date +%s)-t0)/60 )) min"
