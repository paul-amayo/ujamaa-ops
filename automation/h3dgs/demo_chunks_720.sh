#!/bin/bash
# demo_chunks_720.sh <tag> <survey> <model> <chunk> <proj> [script extra] — re-render one reel at 1280x720 (scale 1.0; the
# 09-27 chain default was 0.5 = 640x360, soft on a 1080p gallery screen — dashboard, 10-01 22:3x). Same chain, new dir
# ~/logs/demo_chunks/<tag>_720/. Queued behind the v3 fruit test (quiet-card check, anchored patterns).
set -uo pipefail
TAG=$1; SV=$2; MODEL=$3; CHUNK=$4; PROJ=$5; EXTRA=${6:-}
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new; O=/home/paperspace/logs/demo_chunks/${TAG}_720; mkdir -p $O
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*fruit_wm_test_v3\.sh|^bash [^ ]*sidecar_fruit_densify_chunk\.sh|^[^ ]*python[0-9.]* [^ ]*sidecar_(demo_overlay|row_backdrop)\.py' > /dev/null 2>&1; }
say "$TAG 720p queued: waiting for fruit_wm_test_v3 done, then a quiet card"
until grep -qE 'fruit_wm_test_v3 done' /home/paperspace/logs/fruit_wm_test.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EMB=$(ls -t /home/paperspace/data/citrus_all/$SV/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); t0=$(date +%s)
say "=== $TAG 720p start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
(cd $NS && pixi run python $A/sidecar_demo_path.py --survey $SV --blocks $MODEL --orbit-block $MODEL --orbit-seconds 0 --chunk-drive --sidecar-root experimental/h3dgs_sidecar_chunks --proj $PROJ --fps 8 --scale 1.0 --out $O 2>&1 | grep -aE '^\[path\] drive|Error|Traceback' | tee -a $L)
t1=$(date +%s); $PYH $A/sidecar_row_backdrop.py --survey $SV --blocks $CHUNK --path $O/demo_path.json --proj $PROJ --exposure mean --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\] (chunk|[0-9]+ frames)|Error|Traceback|out of memory' | tee -a $L
say "$TAG 720p backdrop in $(( $(date +%s)-t1 ))s"; [ "$(ls $O/backdrop 2>/dev/null | wc -l)" -gt 100 ] || { say "$TAG 720p: backdrop FAILED"; exit 1; }
t1=$(date +%s); (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --path $O/demo_path.json --models $MODEL --sidecar-root experimental/h3dgs_sidecar_chunks --proj $PROJ --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --min-area 3200 --tree-min-area 800 --fruit-min-area 120 2>&1 | grep -aE '^\[maps\] block|^\[demo\] wrote|Error|Traceback' | tee -a $L)
say "$TAG 720p maps + provisional cut in $(( $(date +%s)-t1 ))s"; [ "$(ls $O/maps/$MODEL 2>/dev/null | wc -l)" -gt 100 ] || { say "$TAG 720p: identity maps FAILED"; exit 1; }
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $SV --path $O/demo_path.json --maps $O/maps --models $MODEL $EXTRA --out $O/demo_script.json 2>&1 | grep -aE '^\[script\] +[0-9]+-|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --path $O/demo_path.json --models $MODEL --sidecar-root experimental/h3dgs_sidecar_chunks --proj $PROJ --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --min-area 3200 --tree-min-area 800 --fruit-min-area 120 --skip-maps --script $O/demo_script.json 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L)
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $O 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
say "$TAG 720p done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
