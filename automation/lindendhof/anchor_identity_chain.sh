#!/bin/bash
# anchor_identity_chain.sh — the Lindendhof identity design for any A300 lane survey (2026-10-12): SAM3 anchor masks (already run by
# trunk_masks.py into $ID/$MASKS) -> LiDAR placement in the anchor band (ZLO..ZHI above ground) -> 1-D density-peak census (rows >= ROW_SEP
# apart, anchors >= MIN_SEP apart) -> per-frame maps (canopy CZLO..CZHI above ground within ROW_HALF of a row -> nearest anchor within
# TREE_HALF; anchor masks override) -> graph embedder -> per-lane v3 census + seed + verdicts.
# Env: S, LANES ("1,2,..."), MASKS, ZLO, ZHI, MIN_SEP, SIGMA, ROW_SEP, ROW_HALF, TREE_HALF, CZLO, CZHI, WSMAP, EN, SITE
set -uo pipefail
: ${S:?} ${LANES:?} ${MASKS:?} ${EN:?}
ID=$S/experimental/identity; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native
LD=/home/paperspace/code/automation/lindendhof; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; EMB=$ID/embedder/$EN/ckpts/model_best.pth
L=/home/paperspace/logs/anchor_identity_$(basename $S).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s)
say "=== anchor identity: $S lanes $LANES masks $MASKS band ${ZLO:-0.2}-${ZHI:-0.7} sep ${MIN_SEP:-1.0} rows ${ROW_SEP:-2.4} canopy ${CZLO:-0.5}-${CZHI:-4.0} widths ${ROW_HALF:-1.2}/${TREE_HALF:-1.0}"
python3 $LD/trunk_obs_lidar.py $S $ID/$MASKS $LANES 2>&1 | grep -a '^\[trunk-obs' | tee -a $L
(cd $NS && pixi run python $LD/tree_census_v2.py $S $ID/$MASKS $ID ${MIN_SEP:-1.0} ${SIGMA:-0.08} 5 8) 2>&1 | grep -a '^\[census-v2\]\|Error\|Traceback' | tee -a $L
[ -e $ID/tree_census.json ] || { say "census FAILED"; exit 1; }
(cd $NS && pixi run python $LD/lindendhof_identity_build.py $S $ID $LANES) 2>&1 | grep -a '^\[build\]\|Error\|Traceback' | tee -a $L
if [ ! -e $EMB ]; then t1=$(date +%s)
  (cd $NS && pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $ID/marker_hierarchy.json --experiment-name $EN --output-dir $ID/embedder \
     --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms) > /home/paperspace/logs/anchor_embedder_$(basename $S).log 2>&1
  say "embedder rc=$? in $(( $(date +%s)-t1 )) s: $(grep -a 'Row (retrieval)' /home/paperspace/logs/anchor_embedder_$(basename $S).log | tail -1 | tr -s ' ')"; fi
[ -e $EMB ] || { say "embedder FAILED"; exit 1; }
for n in $(echo $LANES | tr ',' ' '); do C=$ID/cells/lane$n; WSN=$(python3 -c "import os; m=dict(kv.split(':') for kv in os.environ.get('WSMAP','').split(',') if kv); print(m.get('$n', 'chunk_lane$n'))"); WS=$S/experimental/v3/$WSN
  [ -e $C/text_bank.npz ] || (cd $NS && pixi run python $NT/build_text_bank.py --manifest $C/supervision/trees_only/manifest.json --hierarchy-json $ID/marker_hierarchy.json --out $C/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
  [ -e $C/frames.json ] || python3 $NT/pick_frames.py --names $C/split_names.json --sup $C/supervision/trees_only --must --n-train 4 --n-eval 10 --out $C/frames.json | tee -a $L
  CFG=$WS/outputs/sf_60000_bilateral/splatfacto/run/config.yml CELL=$C EMB=$EMB HJ=$ID/marker_hierarchy.json KF=$WS/images OUT=$WS/identity SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh > /dev/null 2>&1
  say "lane $n: $(grep -a '\[v3 means\]' /home/paperspace/logs/v3_identity_$(basename $S)_$WSN.log 2>/dev/null | tail -1 | cut -c1-300)"
done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
