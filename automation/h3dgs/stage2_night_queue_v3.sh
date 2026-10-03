#!/bin/bash
# stage2_night_queue_v3.sh (2026-10-02 19:5x) — after the lane long-run (lane_v2_lr1e2_7k5, pilot 2 continued to 7,500 iters):
# (a) bg-weight pilot on the lane: lr 1e-2, 2 k iters, high_loss_bg_weight 0.1 (unlabelled pixels pull blended-in gaussians to the
#     zero/void target instead of leaving them free — pilot 2 showed free movement smears the seed);
# (b) decision on the lane verdicts vs the SEED (image_64 TREE 30 0.692; image_40 TREE 18/19 0.027; image_52 TREE 25 0.129):
#     a configuration wins if TREE 30 >= 0.68 AND (TREE 18 or 19 >= 0.127 or TREE 25 >= 0.229); first winner of [7k5, bg] is used;
# (c) citrus cells (05 0_0, 05 1_0 fruit, 01 3_1) at 7,500 iters with the winning configuration — or NIGHT ABORTED, card free.
set -uo pipefail
L=/home/paperspace/logs/stage2_train_chunk.log; NL=/home/paperspace/logs/stage2_night.log; VL=/home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_verdicts.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $NL; }; A=/home/paperspace/logs/stage2_train_chunk_v3.sh
until grep -qaE '^\[[0-9 :-]+\] (=== stage2 live refine lane_v2_lr1e2_7k5 done|lane_v2_lr1e2_7k5: train FAILED)' $L; do sleep 60; done
S=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root; O=$S/experimental/h3dgs_sidecar_chunks/chunk_lane
say "=== night v3: lane long-run finished; bg-weight pilot next; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
O=$O SEED=$O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2_IMG_7993_s0_cabbage_v2_e400/high/2026-10-02_072255/nerfstudio_models/step-000015000.ckpt SUP=$O/supervision/trees_only EMB=$S/prod/bateleur/embedder/IMG_7993_s0_cabbage_v2_e400/ckpts/model_best.pth HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 TAG=lane_v2_lr1e2_bg01_2k ITERS=2000 SAVE=1000 FRAMES="image_64.png image_40.png image_52.png" FRUIT=0 SV=IMG_7993_s0_cabbage LR=0.01 BGW=0.1 bash $A >> $NL 2>&1
iou(){ grep -aE "^\[stage2_train_$1 $2 $3\] $4 " $VL | sed -E 's/ +/ /g' | grep -oE 'IoU [0-9.]+' | head -1 | cut -d' ' -f2; }
WIN=""; WBG=""
for C in "lane_v2_lr1e2_7k5 step5500" "lane_v2_lr1e2_bg01_2k step2000"; do set -- $C; T=$1; ST=$2
  t30=$(iou $T $ST image_64.png 'TREE 30'); t18=$(iou $T $ST image_40.png 'TREE 18'); t19=$(iou $T $ST image_40.png 'TREE 19'); t25=$(iou $T $ST image_52.png 'TREE 25')
  say "decision $T: TREE 30 ${t30:-?} (seed 0.692), 18 ${t18:-?} / 19 ${t19:-?} (seed 0.027), 25 ${t25:-?} (seed 0.129)"
  ok=$(python3 -c "
t30,t18,t19,t25=[float(x) if x else 0.0 for x in ['${t30:-}','${t18:-}','${t19:-}','${t25:-}']]
print(1 if t30>=0.68 and (max(t18,t19)>=0.127 or t25>=0.229) else 0)")
  if [ "$ok" = "1" ] && [ -z "$WIN" ]; then WIN=$T; [ "$T" = "lane_v2_lr1e2_bg01_2k" ] && WBG=0.1; fi
done
[ -n "$WIN" ] || { say "NIGHT ABORTED: no configuration beat the seed on the lane (7.5k at lr 1e-2, bg 0.1 at lr 1e-2) — citrus cells not run; card free"; exit 0; }
say "=== citrus cells with the winning configuration $WIN (lr 0.01, bg weight ${WBG:-0}), 7,500 iters each"
run(){ local SV=$1 DN=$2 CN=$3 SUPN=$4 SEEDG=$5 FRUIT=$6 PROJ=$7 TAG=$8; local S=/home/paperspace/data/citrus_all/$SV; local O=$S/experimental/h3dgs_sidecar_chunks/$DN
  local SEED; SEED=$(ls $O/splat_runs_FEATFIX/$SEEDG/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1); [ -n "$SEED" ] || { say "--- $SV $DN: no seed under $SEEDG — skipped"; return; }
  local EMB; EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
  local FR; FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ 2>>$NL | head -1); local FRS="$FR"
  if [ "$FRUIT" = "1" ]; then local FF; FF=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ --fruit 2>>$NL | head -1); [ -n "$FF" ] && [ "$FF" != "$FR" ] && FRS="$FR $FF"; fi
  [ -n "$FR" ] || { say "--- $SV $DN: no verdict frame from the picker — skipped"; return; }
  say "--- $SV $DN: seed $(echo $SEED | sed "s|$O/splat_runs_FEATFIX/||"), sup $SUPN, frames $FRS"
  O=$O SEED=$SEED SUP=$O/supervision/$SUPN EMB=$EMB HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 TAG=$TAG ITERS=7500 SAVE=2500 FRAMES="$FRS" FRUIT=$FRUIT SV=$SV LR=0.01 BGW=$WBG bash $A >> $NL 2>&1
}
run 05_13D_Jackal chunk_0_0_expo 0_0 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_0_0
run 05_13D_Jackal chunk_1_0_expo 1_0 trees_fruit_v3 stage2_censusinit_fruitdensify_fruit_densify_wm_t0s_r2 1 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_1_0_fruit
run 01_13B_Jackal chunk_3_1 3_1 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/01_13B_Jackal/experimental/h3dgs live_3_1
say "=== night v3 done; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
