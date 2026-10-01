#!/bin/bash
# demo_chunks_1_0_fruit.sh — the v3 fruit side-car (wm_t0s: tail 0, metres gate, split at the gate) put tree 5's oranges at
# IoU 0.301 on the chunk (>= the 0.25 rule), so: (a) re-composite the existing 640x360 05 1_0 reel WITH the fruit segment
# (identity maps reused, fruit maps rendered from the wm_t0s run; the no-fruit cut kept as demo_nofruit.mp4), then (b) the
# full chain at 1280x720 with fruit into 05_1_0_720/, so the morning choice has all three cells at 720p. Waits for the 720p
# queue (01 3_1, 05 0_0) and a quiet card.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new; D=/home/paperspace/logs/demo_chunks
sv=05_13D_Jackal; model=chunk_1_0_expo; chunk=1_0; proj=experimental/h3dgs_expo; EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
FA="--fruit-models $model --fruit-seed-tag fruitdensify_fruit_densify_wm_t0s_r2 --fruit-verdict-tag fruitdensify_fruit_densify_wm_t0s --fruit-max-dist 30"
FS="--fruit-models $model --fruit-count 61 --fruit-tree 5"
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*demo_chunks_720(_queue)?\.sh|^[^ ]*python[0-9.]* [^ ]*sidecar_(demo_overlay|row_backdrop)\.py' > /dev/null 2>&1; }
say "05 1_0 fruit reels queued: waiting for demo_chunks_720_queue done, then a quiet card"
until grep -qE '^\[[0-9 :-]+\] === demo_chunks_720_queue done' $L 2>/dev/null; do sleep 120; done   # anchored on the queue's own end marker: the earlier pattern matched THIS script's 'waiting for … done' line
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (a) 640x360 re-composite with fruit
O=$D/05_1_0; t0=$(date +%s); say "=== 05 1_0 fruit (360p re-composite) start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
[ -e $O/demo_nofruit.mp4 ] || cp $O/demo.mp4 $O/demo_nofruit.mp4
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --reuse-maps $FA 2>&1 | grep -aE '^\[maps|^\[demo\] wrote|Error|Traceback' | tee -a $L)
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --fruit-maps $O/maps_fruit --models $model $FS --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --skip-maps --script $O/demo_script.json $FA 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L)
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $O 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
say "05 1_0 with fruit (360p) done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 (no-fruit cut: demo_nofruit.mp4)"
# (b) 1280x720 full chain with fruit
O=$D/05_1_0_720; mkdir -p $O; t0=$(date +%s); say "=== 05 1_0 fruit 720p start"
(cd $NS && pixi run python $A/sidecar_demo_path.py --survey $sv --blocks $model --orbit-block $model --orbit-seconds 0 --chunk-drive --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --fps 8 --scale 1.0 --out $O 2>&1 | grep -aE '^\[path\] drive|Error|Traceback' | tee -a $L)
$PYH $A/sidecar_row_backdrop.py --survey $sv --blocks $chunk --path $O/demo_path.json --proj $proj --exposure mean --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\] [0-9]+ frames|Error|Traceback|out of memory' | tee -a $L
[ "$(ls $O/backdrop 2>/dev/null | wc -l)" -gt 100 ] || { say "05_1_0 720p: backdrop FAILED"; exit 1; }
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --min-area 3200 --tree-min-area 800 --fruit-min-area 120 $FA 2>&1 | grep -aE '^\[maps|^\[demo\] wrote|Error|Traceback' | tee -a $L)
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --fruit-maps $O/maps_fruit --models $model $FS --out $O/demo_script.json 2>&1 | grep -aE '^\[script\] +[0-9]+-|^\[script\] fruit|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --min-area 3200 --tree-min-area 800 --fruit-min-area 120 --skip-maps --script $O/demo_script.json $FA 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L)
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $O 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
say "05 1_0 with fruit (720p) done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
say "=== demo_chunks_1_0_fruit done"
