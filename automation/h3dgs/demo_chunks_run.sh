#!/bin/bash
# demo_chunks_run.sh — the 27 Sep 'ask the orchard' chain on the Tuesday-demo CHUNKS (Paul, 10-01: "both chunks"):
#   05 chunk_1_0_expo (h3dgs_expo backdrop with the mean trained exposure; fruit segment only if the world-unit re-test is honest)
#   01 chunk_3_1      (no fruit; 'show me the trees at the end of row N')
# Per chunk: backdrop (h3dgs env, compact hierarchy renderer) -> identity maps + provisional cut (nerf_new, HiGH side-car)
# -> question script from the maps -> final composite + mp4 (maps reused). Waits for the fruit re-test to finish and for a
# quiet card (anchored trainer patterns, < 8 GB for 2 consecutive minutes). Outputs ~/logs/demo_chunks/<chunk>/demo.mp4.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
D=/home/paperspace/logs/demo_chunks; FRUIT_MIN_IOU=${FRUIT_MIN_IOU:-0.25}
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*fruit_wm_test\.sh|^bash [^ ]*sidecar_fruit_densify_chunk\.sh' > /dev/null 2>&1; }
say "queued: waiting for the fruit re-test to finish, then a quiet card"
until grep -qE 'fruit_wm_test done' /home/paperspace/logs/fruit_wm_test.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
say "=== start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
# fruit decision from the world-unit re-test's verdict (tree 5 "wildly" on chunk_1_0_expo)
WM=$(grep -aE '^\[chunk_1_0_expo sidecar fruitdensify_fruit_densify_wm ' /home/paperspace/logs/sidecar_05_13D_Jackal_fruit_verdicts.log 2>/dev/null | grep -aoE 'FRUIT of 5 +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | grep -oE 'IoU [0-9.]+' | awk '{print $2}' | sort -g | tail -1)
FRUIT_ARGS=""; FRUIT_SCRIPT=""
if [ -n "${WM:-}" ] && awk -v v="$WM" -v m="$FRUIT_MIN_IOU" 'BEGIN{exit !(v>=m)}'; then
  FRUIT_ARGS="--fruit-models chunk_1_0_expo --fruit-seed-tag fruitdensify_fruit_densify_wm_r2 --fruit-verdict-tag fruitdensify_fruit_densify_wm --fruit-max-dist 30"; FRUIT_SCRIPT="--fruit-models chunk_1_0_expo --fruit-count 61 --fruit-tree 5"
  say "fruit: world-unit re-test tree-5 IoU $WM >= $FRUIT_MIN_IOU -> fruit segment ON (wm side-car)"
else say "fruit: world-unit re-test tree-5 IoU ${WM:-none} < $FRUIT_MIN_IOU -> NO fruit segment (Adinkra gives the count)"; fi
run_chunk(){ local sv=$1 model=$2 chunk=$3 proj=$4 tag=$5 extra_script=$6 fruit_args=$7 fruit_script=$8
  local O=$D/$tag; mkdir -p $O; local EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); local t0=$(date +%s)
  say "--- $tag: backdrop through $proj (exposure mean)"
  $PYH $A/sidecar_row_backdrop.py --survey $sv --blocks $chunk --path $O/demo_path.json --proj $proj --exposure mean --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\]|Error|Traceback' | tee -a $L
  [ "$(ls $O/backdrop 2>/dev/null | wc -l)" -gt 100 ] || { say "$tag: backdrop FAILED"; return 1; }
  say "--- $tag: identity maps + provisional cut"
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 $fruit_args 2>&1 | grep -aE '^\[maps|^\[demo\]|Error|Traceback' | tee -a $L)
  [ "$(ls $O/maps/$model 2>/dev/null | wc -l)" -gt 100 ] || { say "$tag: identity maps FAILED"; return 1; }
  say "--- $tag: question script"
  (cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --fruit-maps $O/maps_fruit --models $model $fruit_script $extra_script --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
  say "--- $tag: final composite"
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --skip-maps --script $O/demo_script.json $fruit_args 2>&1 | grep -aE '^\[demo\]|Error|Traceback' | tee -a $L)
  say "$tag done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
}
run_chunk 05_13D_Jackal chunk_1_0_expo 1_0 experimental/h3dgs_expo 05_1_0 "" "$FRUIT_ARGS" "$FRUIT_SCRIPT"
run_chunk 01_13B_Jackal chunk_3_1 3_1 experimental/h3dgs 01_3_1 "--end-of-row 22" "" ""
say "=== demo_chunks_run done"
