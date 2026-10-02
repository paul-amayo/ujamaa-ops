#!/bin/bash
# demo_gwakungu_run.sh — cabbage query reel on IMG_7993_s0 (h3dgs_8m, chunk 'lane'): side-car first (GPU ~30 min), then the
# chunk-mode reel at 540x960 portrait, 5 fps, questions with the plant noun 'cabbage' and the registry count (29 in 2 rows).
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
SV=IMG_7993_s0_cabbage; R=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root; O=/home/paperspace/logs/demo_chunks/gwakungu_7993; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "=== gwakungu cabbage: side-car start (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
bash $A/sidecar_phone_chunk.sh $SV $R lane 6 2>&1 | grep -aE 'ready in|failed|FAILED|missing|skipping|bootstrap in|census in|sidecar\]|init0|background|floors|no bootstrap' | tee -a $L
ls $R/experimental/h3dgs_sidecar_chunks/chunk_lane/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1 || { say "gwakungu: side-car FAILED"; exit 1; }
EMB=$(ls -t $R/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); t0=$(date +%s)
say "--- gwakungu: reel (path 540x960, backdrop, maps, script, composite)"
(cd $NS && pixi run python $A/sidecar_demo_path.py --survey $SV --survey-root $R --blocks chunk_lane --orbit-block chunk_lane --orbit-seconds 0 --chunk-drive --chunk lane --min-pass 20 --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --fps 8 --scale 0.5 --out $O 2>&1 | grep -aE '^\[path\]|Error|Traceback' | tee -a $L)
$PYH $A/sidecar_row_backdrop.py --survey $SV --survey-root $R --blocks lane --path $O/demo_path.json --proj experimental/h3dgs --exposure mean --out $O/backdrop 2>&1 | grep -aE '^\[backdrop\]|Error|Traceback|out of memory' | tee -a $L
[ "$(ls $O/backdrop 2>/dev/null | wc -l)" -gt 20 ] || { say "gwakungu: backdrop FAILED"; exit 1; }
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --survey-root $R --path $O/demo_path.json --models chunk_lane --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --backdrop-dir $O/backdrop --out $O --fps 5 --max-dist 60 --noun cabbage --min-area 400 --tree-min-area 100 2>&1 | grep -aE '^\[maps|^\[demo\]|Error|Traceback' | tee -a $L)
[ "$(ls $O/maps/chunk_lane 2>/dev/null | wc -l)" -gt 20 ] || { say "gwakungu: identity maps FAILED"; exit 1; }
(cd $NS && pixi run python $A/sidecar_demo_script_chunk.py --survey $SV --survey-root $R --path $O/demo_path.json --maps $O/maps --models chunk_lane --noun cabbage --ask-count --hold 10 --out $O/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L)
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --survey-root $R --path $O/demo_path.json --models chunk_lane --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --backdrop-dir $O/backdrop --out $O --fps 5 --max-dist 60 --noun cabbage --min-area 400 --tree-min-area 100 --skip-maps --script $O/demo_script.json 2>&1 | grep -aE '^\[demo\]|Error|Traceback' | tee -a $L)
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $O --cols 4 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
say "gwakungu reel done in $(( ($(date +%s)-t0)/60 )) min -> $O/demo.mp4 ($(du -h $O/demo.mp4 2>/dev/null | cut -f1))"
say "=== demo_gwakungu_run done"
