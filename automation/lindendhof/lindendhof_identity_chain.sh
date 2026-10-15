#!/bin/bash
# Lindendhof identity chain after the census (2026-10-12): supervision for lanes 2-5 (lane 1 built and checked by eye), the graph
# embedder on the 126-tree / 7-row hierarchy (citrus graph recipe), then per lane: text bank, verdict frames, v3 census + seed + verdicts.
set -uo pipefail
S=/home/paperspace/data/lindendhof/nov_2025_a300; ID=$S/experimental/identity; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3
NT=/home/paperspace/code/automation/h3dgs/native; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; EN=nov_2025_a300_trunks_v1; EMB=$ID/embedder/$EN/ckpts/model_best.pth
L=/home/paperspace/logs/lindendhof_identity_chain.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s)
say "=== Lindendhof identity chain"
(cd $NS && pixi run python /home/paperspace/code/automation/lindendhof/lindendhof_identity_build.py $S $ID 2,3,4,5) 2>&1 | grep -a '^\[build\]\|Error\|Traceback' | tee -a $L
if [ ! -e $EMB ]; then t1=$(date +%s)
  (cd $NS && pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $ID/marker_hierarchy.json --experiment-name $EN --output-dir $ID/embedder \
     --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms) > /home/paperspace/logs/lindendhof_embedder.log 2>&1
  say "embedder rc=$? in $(( $(date +%s)-t1 )) s: $(ls $EMB 2>/dev/null || echo NO model_best)"; fi
[ -e $EMB ] || { say "embedder FAILED (logs/lindendhof_embedder.log)"; exit 1; }
for n in 1 2 3 4 5; do C=$ID/cells/lane$n; WS=$S/experimental/v3/chunk_lane$n
  [ -e $C/text_bank.npz ] || (cd $NS && pixi run python $NT/build_text_bank.py --manifest $C/supervision/trees_only/manifest.json --hierarchy-json $ID/marker_hierarchy.json --out $C/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
  [ -e $C/frames.json ] || python3 $NT/pick_frames.py --names $C/split_names.json --sup $C/supervision/trees_only --must --n-train 4 --n-eval 10 --out $C/frames.json | tee -a $L
  CFG=$WS/outputs/sf_60000_bilateral/splatfacto/run/config.yml CELL=$C EMB=$EMB HJ=$ID/marker_hierarchy.json KF=$WS/images OUT=$WS/identity SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh > /dev/null 2>&1
  say "lane $n: $(grep -a '\[v3 means\]' /home/paperspace/logs/v3_identity_nov_2025_a300_chunk_lane$n.log 2>/dev/null | tail -1 | cut -c1-300)"
done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
