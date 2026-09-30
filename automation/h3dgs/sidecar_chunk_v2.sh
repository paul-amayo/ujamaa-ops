#!/bin/bash
# sidecar_chunk_v2.sh <survey id> <chunk> [margin m=6] — CHUNK-level containment side-car, v2 (2026-09-30).
# Same chain as sidecar_chunk.sh (dataset from the chunk's poses + supervision -> H3DGS leaves in the cell as a stage-1
# ckpt -> zero features -> 1-iteration bootstrap -> census with the void row -> ratio-2 seed -> verdict), with the inputs
# that were hard-coded made overridable so a chunk from another H3DGS project (05 h3dgs_expo) or another survey (01) runs:
#   SIDECAR_PROJ      H3DGS project dir            (default <survey>/experimental/h3dgs)
#   SIDECAR_HIER      hierarchy (.hier | .hier_opt) (default $SIDECAR_PROJ/output/merged.hier; a chunk's hierarchy.hier_opt works)
#   SIDECAR_TAG       output-dir suffix            (default none -> chunk_<c>; "_expo" -> chunk_<c>_expo, so the skip guard
#                                                   on an existing survey-merge side-car does not fire)
#   SIDECAR_TEMPLATE  stage-1 template ckpt        (default the survey's block_020 side-car ckpt; 01 has none -> pass 05's)
#   SIDECAR_KF        keyframe images for the verdict (default <survey>/prod/scratch_sam3)
# Adds a held-out PSNR of the side-car's own RGB (ns-eval on the dataset's every-10th split) so it can sit next to the
# H3DGS number. Output: <survey>/experimental/h3dgs_sidecar_chunks/chunk_<c><tag>/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/<ts>/
set -uo pipefail
SV=${1:?survey}; CN=${2:?chunk}; MARGIN=${3:-6}; TAG=${SIDECAR_TAG:-}
S=/home/paperspace/data/citrus_all/$SV; P=${SIDECAR_PROJ:-$S/experimental/h3dgs}; HIER=${SIDECAR_HIER:-$P/output/merged.hier}
O=$S/experimental/h3dgs_sidecar_chunks/chunk_${CN}${TAG}; SUP=$O/supervision/trees_only
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=${SIDECAR_KF:-$S/prod/scratch_sam3}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_chunks.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
TEMPLATE=${SIDECAR_TEMPLATE:-$(ls $S/experimental/h3dgs_sidecar/block_020/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1)}
[ -n "$TEMPLATE" ] && [ -e "$TEMPLATE" ] || { say "chunk $CN$TAG: no template ckpt (SIDECAR_TEMPLATE)"; exit 1; }
[ -e "$HIER" ] || { say "chunk $CN$TAG: hierarchy $HIER missing"; exit 1; }
[ -e "$HJ" ] && [ -n "$EMB" ] || { say "chunk $CN$TAG: missing marker_hierarchy.json or embedder"; exit 1; }
if ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1; then say "chunk $CN$TAG: seed exists, skipping"; exit 0; fi
t0=$(date +%s); say "=== chunk $CN$TAG side-car v2 (margin $MARGIN m) | project $P | hierarchy $(basename $HIER) $(du -h $HIER | cut -f1) | template $(basename $(dirname $(dirname $(dirname $TEMPLATE))))"
# 1. dataset + dataparser frame (chunk poses come from the survey project's chunk dir; an expo project's chunks are hard-linked copies of it)
$PYH /home/paperspace/logs/sidecar_chunk_dataset.py $SV $CN $O 2>&1 | grep -a '^\[chunk-dataset\]' | tee -a $L
python3 /home/paperspace/logs/sidecar_chunk_manifest.py $SV $O 2>&1 | grep -a '^\[chunk-manifest\]' | tee -a $L   # trees_only/manifest.json = the owning blocks' word_table union (render_service resolves "tree N" through it)
S1D=$O/splat_runs_STAGE1/stage1_bg00_glref/high/h3dgs_sidecar; mkdir -p $S1D/nerfstudio_models
(cd $NS && pixi run python /home/paperspace/logs/sidecar_dataparser.py $O $S1D 2>&1 | grep -a '^\[dataparser\]' | tee -a $L)
[ -e $S1D/dataparser_transforms.json ] || { say "chunk $CN$TAG: dataparser failed"; exit 1; }
# 2. leaves in the cell -> stage-1 ckpt, + zero features = stage2_init
$PYH /home/paperspace/logs/h3dgs_to_stage1.py --block $O --h3dgs $P --hier $HIER --template $TEMPLATE --dataparser $S1D/dataparser_transforms.json \
  --out $S1D/nerfstudio_models/step-000015000.ckpt --margin $MARGIN --cell $P/camera_calibration/chunks/$CN 2>&1 | grep -a '^\[sidecar\]' | cut -c1-260 | tee -a $L
[ -e $S1D/nerfstudio_models/step-000015000.ckpt ] || { say "chunk $CN$TAG: conversion failed"; exit 1; }
mkdir -p $O/stage2_init_glref/nerfstudio_models
$PYH - $S1D/nerfstudio_models/step-000015000.ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); k = [k for k in ck['pipeline'] if k.endswith('gauss_params.means')][0]
ck['pipeline'][k.replace('means', 'high_features')] = torch.zeros((ck['pipeline'][k].shape[0], 32)); torch.save(ck, sys.argv[2]); print(f"[init0] {ck['pipeline'][k].shape[0]} gaussians + zero features", flush=True)
PY
# 3. bootstrap (config on this geometry; 1 iteration, geometry frozen)
cd $NS; t1=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name stage2_bootstrap_glref \
  --load-dir $O/stage2_init_glref/nerfstudio_models --pipeline.model.freeze-geometry True --pipeline.model.high-loss-weight 1.0 --pipeline.datamanager.semantic-dir $SUP \
  --pipeline.model.rasterize-mode antialiased --pipeline.model.sky-loss-lambda 1.0 --max-num-iterations 1 --steps-per-save 19998 --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 \
  > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}${TAG}_bootstrap.log 2>&1 || true
BOOT=$(ls -t $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/config.yml 2>/dev/null | head -1); [ -n "$BOOT" ] || { say "chunk $CN$TAG: no bootstrap config"; exit 1; }
say "chunk $CN$TAG bootstrap in $(( $(date +%s)-t1 ))s"
# 4. one census with the void row, ratio-2 seed
t1=$(date +%s); W=$O/splat_runs_FEATFIX/interaction_W_glref_bg.npz
CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$BOOT" --supervision-dir $SUP --out-npz $W > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}${TAG}_census.log 2>&1
[ -e $W ] || { say "chunk $CN$TAG: census failed"; exit 1; }; say "chunk $CN$TAG census in $(( $(date +%s)-t1 ))s"
INIT=$O/stage2_init_census_glref_bg_f1.0_r2/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt --dst-dir $INIT --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'background|floors|assigned' | tee -a $L
TS=$(basename $(dirname $BOOT)); RUN=$O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/$TS; mkdir -p $RUN/nerfstudio_models
mv $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_glref_bg_f1.0_r2|" $BOOT > $RUN/config.yml; cp $(dirname $BOOT)/dataparser_transforms.json $RUN/ 2>/dev/null || true
$PYH -c "
import torch,sys,glob; p=glob.glob(sys.argv[1]+'/*.ckpt')[0]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['optimizers']={}; ck['schedulers']={}; torch.save(ck,p)" "$RUN/nerfstudio_models"
rm -rf $O/splat_runs_STAGE1 $O/stage2_init_glref $O/stage2_init_census_glref_bg_f1.0_r2 $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/nerfstudio_models*
# 5. verdict on the top-supervised frame whose camera is INSIDE the cell (rationale in sidecar_chunk_frame.py)
FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $P 2>>$L | head -1)
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/chunk_${CN}${TAG}_bg_f1.0_r2_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[chunk_$CN$TAG sidecar bg_f1.0_r2 $FR] /" >> $VL
rm -rf $O/clip_cache_*
# 6. the side-car's own RGB quality on the dataset's held-out split (every 10th camera), to sit next to the H3DGS number
t1=$(date +%s); timeout 1800 pixi run ns-eval --load-config $RUN/config.yml --output-path $O/psnr_sidecar_$TS.json > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}${TAG}_nseval.log 2>&1 \
  && say "chunk $CN$TAG side-car RGB held-out (ns-eval, $(( $(date +%s)-t1 ))s): $(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; print(' '.join(f'{k} {v:.3f}' for k,v in r.items() if k in ('psnr','ssim','lpips')))" $O/psnr_sidecar_$TS.json)" \
  || say "chunk $CN$TAG side-car RGB held-out: ns-eval failed (see sidecar_${SV}_chunk_${CN}${TAG}_nseval.log)"
rm -rf $O/clip_cache_*
say "chunk $CN$TAG side-car ready in $(( ($(date +%s)-t0)/60 )) min; kept $(du -sh $O | cut -f1) ($(df --output=avail -BG / | tail -1 | tr -dc 0-9)G free); run $RUN; verdict $FR: $(grep -aE "^\[chunk_$CN$TAG sidecar" $VL | grep -oE '(TREE|ROW) of [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
