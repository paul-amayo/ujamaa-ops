#!/bin/bash
# stage2_night_queue_v4.sh (2026-10-02 21:5x) — after the relaunched bg-weight pilot (lane_v2_lr1e2_bg01_2k): apply the night's
# rule (image_64 TREE 30 >= 0.68 AND (image_40 TREE 18 or 19 >= 0.127 or image_52 TREE 25 >= 0.229), seed 0.692 / 0.027 / 0.129);
# if it wins AND >= 20 GB is free, run the citrus cells at lr 1e-2 + bg 0.1, 7,500 iters, ONE checkpoint each (disk at 99 %).
set -uo pipefail
L=/home/paperspace/logs/stage2_train_chunk.log; NL=/home/paperspace/logs/stage2_night.log; VL=/home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_verdicts.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $NL; }; A=/home/paperspace/logs/stage2_train_chunk_v3.sh
until [ "$(grep -caE '^\[[0-9 :-]+\] (=== stage2 live refine lane_v2_lr1e2_bg01_2k done|lane_v2_lr1e2_bg01_2k: train FAILED)' $L)" -ge 2 ]; do sleep 60; done
if grep -aE '^\[[0-9 :-]+\] (=== stage2 live refine lane_v2_lr1e2_bg01_2k done|lane_v2_lr1e2_bg01_2k: train FAILED)' $L | tail -1 | grep -q FAILED; then say "NIGHT (bg rerun) FAILED — see stage2_train_IMG_7993_s0_cabbage_lane_v2_lr1e2_bg01_2k.log; nothing launched"; exit 1; fi
iou(){ grep -aE "^\[stage2_train_$1 $2 $3\] $4 " $VL | tail -1 | sed -E 's/ +/ /g' | grep -oE 'IoU [0-9.]+' | head -1 | cut -d' ' -f2; }
T=lane_v2_lr1e2_bg01_2k; ST=step2000; t30=$(iou $T $ST image_64.png 'TREE 30'); t18=$(iou $T $ST image_40.png 'TREE 18'); t19=$(iou $T $ST image_40.png 'TREE 19'); t25=$(iou $T $ST image_52.png 'TREE 25')
say "decision $T: TREE 30 ${t30:-?} (seed 0.692), 18 ${t18:-?} / 19 ${t19:-?} (seed 0.027), 25 ${t25:-?} (seed 0.129)"
ok=$(python3 -c "
t30,t18,t19,t25=[float(x) if x else 0.0 for x in ['${t30:-}','${t18:-}','${t19:-}','${t25:-}']]
print(1 if t30>=0.68 and (max(t18,t19)>=0.127 or t25>=0.229) else 0)")
[ "$ok" = "1" ] || { say "NIGHT ABORTED (bg 0.1): the bg-weight refine did not beat the seed on the lane either — citrus cells not run; card free"; exit 0; }
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$FREE" -ge 20 ] || { say "bg 0.1 WON on the lane but only ${FREE} GB free (need 20) — citrus cells not run; ask Paul to free the negative checkpoints"; exit 0; }
say "=== citrus cells with lr 0.01 + bg 0.1, 7,500 iters, one checkpoint each (${FREE} GB free)"
run(){ local SV=$1 DN=$2 CN=$3 SUPN=$4 SEEDG=$5 FRUIT=$6 PROJ=$7 TAG=$8; local S=/home/paperspace/data/citrus_all/$SV; local O=$S/experimental/h3dgs_sidecar_chunks/$DN
  local F; F=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$F" -ge 8 ] || { say "--- $SV $DN: skipped, only ${F} GB free"; return; }
  local SEED; SEED=$(ls $O/splat_runs_FEATFIX/$SEEDG/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1); [ -n "$SEED" ] || { say "--- $SV $DN: no seed under $SEEDG — skipped"; return; }
  local EMB; EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
  local FR; FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ 2>>$NL | head -1); local FRS="$FR"
  if [ "$FRUIT" = "1" ]; then local FF; FF=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ --fruit 2>>$NL | head -1); [ -n "$FF" ] && [ "$FF" != "$FR" ] && FRS="$FR $FF"; fi
  [ -n "$FR" ] || { say "--- $SV $DN: no verdict frame from the picker — skipped"; return; }
  say "--- $SV $DN: seed $(echo $SEED | sed "s|$O/splat_runs_FEATFIX/||"), sup $SUPN, frames $FRS"
  O=$O SEED=$SEED SUP=$O/supervision/$SUPN EMB=$EMB HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 TAG=$TAG ITERS=7500 SAVE=7500 FRAMES="$FRS" FRUIT=$FRUIT SV=$SV LR=0.01 BGW=0.1 bash $A >> $NL 2>&1
}
run 05_13D_Jackal chunk_0_0_expo 0_0 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_0_0
run 05_13D_Jackal chunk_1_0_expo 1_0 trees_fruit_v3 stage2_censusinit_fruitdensify_fruit_densify_wm_t0s_r2 1 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_1_0_fruit
run 01_13B_Jackal chunk_3_1 3_1 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/01_13B_Jackal/experimental/h3dgs live_3_1
say "=== night v4 done; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader); $(df --output=avail -BG / | tail -1 | tr -dc 0-9) GB free"
