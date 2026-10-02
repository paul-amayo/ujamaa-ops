#!/bin/bash
# stage2_night_queue.sh (2026-10-02 night) — chained behind the cabbage-lane pilot (stage2_train_chunk.sh TAG=lane_v2_live2k).
# Decision rule (no hit and hope): the night runs ONLY if the pilot trained (rc 0) and its features actually moved — at least
# 60,000 of the 127,827 seeded gaussians displaced by > 0.01 (the August fw2 refine moved 0.00000 = dead optimizer). Budget per
# cell = ~2 h of training from the pilot's measured it/s, clamped to 3k..10k iterations. Then, in order: 05 chunk_0_0_expo
# (Citrus B drive cell, identity), 05 chunk_1_0_expo from the v3 fruit seed (identity + fruit), 01 chunk_3_1 (Citrus A hero),
# cabbage lane at the full budget. Verdicts after each on the chain's own frames (same picker, same train split as the seeds).
# Serving roots are NOT touched — Paul flips them after reading the verdicts.
set -uo pipefail
L=/home/paperspace/logs/stage2_train_chunk.log; NL=/home/paperspace/logs/stage2_night.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $NL; }
A=/home/paperspace/logs/stage2_train_chunk.sh
until grep -qaE '^\[[0-9 :-]+\] (=== stage2 live refine lane_v2_live2k done|lane_v2_live2k: train FAILED)' $L; do sleep 60; done
if grep -qaE 'lane_v2_live2k: train FAILED' $L; then say "NIGHT ABORTED: pilot training failed"; exit 1; fi
MOVED=$(grep -aoE 'rows moved >0.01: [0-9,]+' $L | tail -1 | grep -oE '[0-9,]+$' | tr -d ,)
ITS=$(grep -aoE 'lane_v2_live2k trained in [0-9]+ min \([0-9.]+ it/s\)' $L | tail -1 | grep -oE '[0-9.]+ it/s' | cut -d' ' -f1)
[ "${MOVED:-0}" -ge 60000 ] || { say "NIGHT ABORTED: pilot moved only ${MOVED:-0} seeded rows by >0.01 — the optimizer is dead again; nothing launched"; exit 1; }
ITERS=$(python3 -c "print(max(3000, min(10000, int(round(float('${ITS:-1}')*7200/500))*500)))")
say "=== night queue: pilot moved $MOVED rows at ${ITS:-?} it/s -> $ITERS iters per cell; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
run(){ # SV DN CN SUPNAME SEEDGLOB FRUIT PROJ TAG
  local SV=$1 DN=$2 CN=$3 SUPN=$4 SEEDG=$5 FRUIT=$6 PROJ=$7 TAG=$8; local S=/home/paperspace/data/citrus_all/$SV; local O=$S/experimental/h3dgs_sidecar_chunks/$DN
  local SEED; SEED=$(ls $O/splat_runs_FEATFIX/$SEEDG/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1); [ -n "$SEED" ] || { say "--- $SV $DN: no seed under $SEEDG — skipped"; return; }
  local EMB; EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
  local FR; FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ 2>>$NL | head -1); local FRS="$FR"
  if [ "$FRUIT" = "1" ]; then local FF; FF=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $PROJ --fruit 2>>$NL | head -1); [ -n "$FF" ] && [ "$FF" != "$FR" ] && FRS="$FR $FF"; fi
  [ -n "$FR" ] || { say "--- $SV $DN: no verdict frame from the picker — skipped"; return; }
  say "--- $SV $DN: seed $(echo $SEED | sed "s|$O/splat_runs_FEATFIX/||"), sup $SUPN, frames $FRS, $ITERS iters"
  O=$O SEED=$SEED SUP=$O/supervision/$SUPN EMB=$EMB HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 TAG=$TAG ITERS=$ITERS SAVE=2500 FRAMES="$FRS" FRUIT=$FRUIT SV=$SV bash $A >> $NL 2>&1
}
run 05_13D_Jackal chunk_0_0_expo 0_0 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_0_0
run 05_13D_Jackal chunk_1_0_expo 1_0 trees_fruit_v3 stage2_censusinit_fruitdensify_fruit_densify_wm_t0s_r2 1 /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo live_1_0_fruit
run 01_13B_Jackal chunk_3_1 3_1 trees_only stage2_censusinit_glref_bg_f1.0_r2 0 /home/paperspace/data/citrus_all/01_13B_Jackal/experimental/h3dgs live_3_1
S=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root; O=$S/experimental/h3dgs_sidecar_chunks/chunk_lane
say "--- cabbage lane: full budget $ITERS iters from the v2 seed"
O=$O SEED=$O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2_IMG_7993_s0_cabbage_v2_e400/high/2026-10-02_072255/nerfstudio_models/step-000015000.ckpt SUP=$O/supervision/trees_only EMB=$S/prod/bateleur/embedder/IMG_7993_s0_cabbage_v2_e400/ckpts/model_best.pth HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 TAG=lane_v2_live ITERS=$ITERS SAVE=2500 FRAMES="image_33.png image_40.png image_52.png image_64.png" FRUIT=0 SV=IMG_7993_s0_cabbage bash $A >> $NL 2>&1
say "=== night queue done; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
