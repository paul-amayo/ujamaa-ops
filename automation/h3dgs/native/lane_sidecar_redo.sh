#!/bin/bash
# lane_sidecar_redo.sh — the cabbage lane's side-car dump OOM'd beside the 01 census (15:09); redo it when >= 14 GB of
# GPU is free, then score the side-car maps and rebuild the lane table (UJAMAA, 2026-10-03).
set -uo pipefail
G=/home/paperspace/data/image_farm/gwakungu/2026-05-16; D=$G/IMG_7993_s0_cabbage/demo_root; SC=$D/experimental/h3dgs_sidecar_chunks/chunk_lane
N=$D/experimental/h3dgs_native/chunk_lane; NT=/home/paperspace/code/automation/h3dgs/native; NS=/home/paperspace/code/nerf_new
EMB=$D/prod/bateleur/embedder/IMG_7993_s0_cabbage_v2_e400/ckpts/model_best.pth; HJ=$D/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$D/prod/scratch_sam3
CFG=$(ls $SC/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2_IMG_7993_s0_cabbage_v2_e400/high/*/config.yml | head -1)
L=/home/paperspace/logs/native_census_lane_redo.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 14000 ]; do sleep 20; done
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
say "lane side-car dump redo ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free)"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $NT/sidecar_feature_dump.py --config $CFG --frames $FR --out-dir $N/maps_sidecar --tag sidecar) > $L.dump 2>&1
grep -aE '^\[dump\]|OutOfMemory|Traceback' $L.dump | sed -E 's/ -> .*//' | tee -a $L
sed -i '/^\[sidecar [^]]*\] MISSING/d' $N/verdicts.log
bash $NT/score_maps.sh $N sidecar maps_sidecar/%s_sidecar.npz $SC/supervision/trees_only $EMB $HJ $KF 4 | tee -a $L
python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json --canonical /home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_verdicts.log --canonical-tag 'chunk_lane sidecar glref_bg_f1.0_r2_IMG_7993_s0_cabbage_v2_e400' > $N/table.md
sed -n '/| kind/,$p' $N/table.md | tee -a $L; say "lane redo done"
