#!/bin/bash
# demo_chunks_01_retry.sh — 01 chunk_3_1 video retry (the first backdrop pass OOM'd after 43 frames alone on the card: the
# torch caching allocator starved the hierarchy's C++ expand_to_size; the backdrop now empties the cache every 25 frames and
# runs with expandable segments; tau 3 fallback if it OOMs again). Also: if the corrected fruit test (wm_t0) gives tree 5 an
# honest verdict (IoU >= 0.25), re-composite the 05 video WITH the fruit segment (maps reused, fruit maps from the wm_t0 run).
# Waits for fruit_wm_test_v2 to finish and a quiet card.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new; D=/home/paperspace/logs/demo_chunks
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*fruit_wm_test_v2\.sh|^bash [^ ]*sidecar_fruit_densify_chunk\.sh' > /dev/null 2>&1; }
say "01 retry queued: waiting for fruit_wm_test_v2 done, then a quiet card"
until grep -qE 'fruit_wm_test_v2 done' /home/paperspace/logs/fruit_wm_test.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
say "=== 01 retry start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sv=01_13B_Jackal; model=chunk_3_1; chunk=3_1; proj=experimental/h3dgs; O=$D/01_3_1; EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); t0=$(date +%s)
for TAU in 0 3; do
  say "--- 01_3_1: backdrop (tau $TAU, exposure mean, cache emptied every 25 frames)"
  $PYH $A/sidecar_row_backdrop.py --survey $sv --blocks $chunk --path $O/demo_path.json --proj $proj --exposure mean --tau $TAU --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\]|Error|Traceback|out of memory' | tee -a $L
  [ "$(ls $O/backdrop 2>/dev/null | wc -l)" -ge 447 ] && break
  say "01_3_1: backdrop incomplete at tau $TAU ($(ls $O/backdrop 2>/dev/null | wc -l)/447 frames)"
done
[ "$(ls $O/backdrop 2>/dev/null | wc -l)" -ge 447 ] || { say "01_3_1: backdrop FAILED at both taus"; exit 1; }
say "--- 01_3_1: identity maps + provisional cut"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 2>&1 | grep -aE '^\[maps|^\[demo\]|Error|Traceback' | tee -a $L)
[ "$(ls $O/maps/$model 2>/dev/null | wc -l)" -gt 100 ] || { say "01_3_1: identity maps FAILED"; exit 1; }
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --models $model --end-of-row 22 --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --skip-maps --script $O/demo_script.json 2>&1 | grep -aE '^\[demo\]|Error|Traceback' | tee -a $L)
say "01_3_1 done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
# 05 with fruit, only if the corrected test is honest
WM=$(grep -aE '^\[chunk_1_0_expo sidecar fruitdensify_fruit_densify_wm_t0 ' /home/paperspace/logs/sidecar_05_13D_Jackal_fruit_verdicts.log 2>/dev/null | grep -aoE 'FRUIT of 5 +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | grep -oE 'IoU [0-9.]+' | awk '{print $2}' | sort -g | tail -1)
if [ -n "${WM:-}" ] && awk -v v="$WM" 'BEGIN{exit !(v>=0.25)}'; then
  say "--- 05_1_0: fruit verdict (wm_t0) tree-5 IoU $WM >= 0.25 -> re-composite WITH the fruit segment"
  sv=05_13D_Jackal; model=chunk_1_0_expo; proj=experimental/h3dgs_expo; O=$D/05_1_0; EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); t0=$(date +%s)
  FA="--fruit-models chunk_1_0_expo --fruit-seed-tag fruitdensify_fruit_densify_wm_t0_r2 --fruit-verdict-tag fruitdensify_fruit_densify_wm_t0 --fruit-max-dist 30"
  cp $O/demo.mp4 $O/demo_nofruit.mp4
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --reuse-maps $FA 2>&1 | grep -aE '^\[maps|^\[demo\]|Error|Traceback' | tee -a $L)
  (cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --fruit-maps $O/maps_fruit --models $model --fruit-models $model --fruit-count 61 --fruit-tree 5 --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --skip-maps --script $O/demo_script.json $FA 2>&1 | grep -aE '^\[demo\]|Error|Traceback' | tee -a $L)
  say "05_1_0 with fruit done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 (no-fruit cut kept as demo_nofruit.mp4)"
else say "05_1_0: fruit verdict (wm_t0) tree-5 IoU ${WM:-none} < 0.25 -> the no-fruit cut stands"; fi
say "=== demo_chunks_01_retry done"
