#!/bin/bash
# demo_chunks_00.sh — the same 'ask the orchard' cut on 05 chunk_0_0_expo (10-01 21:4x): cell 1_0's in-cell cameras are
# headland / edge passes (the eastern turnaround of the survey), so its video looks AT row ends and open ground; cell 0_0
# holds the oak/pine lane drives of the 09-27 demo. Rendered so Paul can choose in the morning. No fruit. Queued behind the
# 01 retry (same card), quiet-card check as the others.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new; D=/home/paperspace/logs/demo_chunks
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*fruit_wm_test_v2\.sh|^bash [^ ]*demo_chunks_01_retry\.sh|^bash [^ ]*sidecar_fruit_densify_chunk\.sh' > /dev/null 2>&1; }
say "05 0_0 cut queued: waiting for demo_chunks_01_retry done, then a quiet card"
until grep -qE 'demo_chunks_01_retry done' /home/paperspace/logs/demo_chunks_run.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sv=05_13D_Jackal; model=chunk_0_0_expo; chunk=0_0; proj=experimental/h3dgs_expo; O=$D/05_0_0; mkdir -p $O; EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); t0=$(date +%s)
say "=== 05 0_0 cut start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
(cd $NS && pixi run python $A/sidecar_demo_path.py --survey $sv --blocks $model --orbit-block $model --orbit-seconds 0 --chunk-drive --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --fps 8 --out $O 2>&1 | grep -aE '^\[path\] drive|Error|Traceback' | tee -a $L)
$PYH $A/sidecar_row_backdrop.py --survey $sv --blocks $chunk --path $O/demo_path.json --proj $proj --exposure mean --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\] (chunk|[0-9]+ frames)|Error|Traceback|out of memory' | tee -a $L
[ "$(ls $O/backdrop 2>/dev/null | wc -l)" -gt 100 ] || { say "05_0_0: backdrop FAILED"; exit 1; }
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 2>&1 | grep -aE '^\[maps\] block|^\[demo\] wrote|Error|Traceback' | tee -a $L)
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $sv --path $O/demo_path.json --maps $O/maps --models $model --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 --skip-maps --script $O/demo_script.json 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L)
say "05_0_0 done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
say "=== demo_chunks_00 done"
