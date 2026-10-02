#!/bin/bash
# sidecar_phone_chunk.sh — containment side-car for a PHONE SEGMENT's single-chunk H3DGS model (Gwakungu cabbages, Paul 10-02:
# "we need cabbage registries and queries otherwise we are just showing renders"). Same chain as sidecar_chunk_v2.sh from the
# leaf conversion on, with the survey layout provided by a shim root (symlinks into the segment: experimental/h3dgs -> the H3DGS
# project, prod/bateleur/scene_graph/marker_hierarchy.json -> the registry, prod/bateleur/embedder/<n>/ckpts/model_best.pth,
# prod/scratch_sam3 -> images) and a side-car dataset already laid out under experimental/h3dgs_sidecar_chunks/chunk_<c>/
# (transforms.json with absolute image paths + supervision/trees_only id maps + manifest.json). No ns-eval (the H3DGS model's
# own PSNR is on record). usage: sidecar_phone_chunk.sh <SV tag for logs> <shim root> <chunk> [margin m=6]
set -uo pipefail
SV=${1:?log tag}; S=${2:?shim root}; CN=${3:?chunk}; MARGIN=${4:-6}
P=$S/experimental/h3dgs; HIER=${SIDECAR_HIER:-$P/output/trained_chunks/$CN/hierarchy.hier_opt}; O=$S/experimental/h3dgs_sidecar_chunks/chunk_${CN}; SUP=$O/supervision/trees_only
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_chunks.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
TEMPLATE=${SIDECAR_TEMPLATE:-$(ls /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar/block_020/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt | head -1)}
for f in "$HIER" "$HJ" "$EMB" "$TEMPLATE" "$O/transforms.json" "$SUP/manifest.json"; do [ -e "$f" ] || { say "chunk $CN: missing $f"; exit 1; }; done
if ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1; then say "chunk $CN: seed exists, skipping"; exit 0; fi
t0=$(date +%s); say "=== phone chunk $CN side-car (margin $MARGIN m) | root $S | hierarchy $(du -h $HIER | cut -f1) | embedder $(basename $(dirname $(dirname $EMB))) | $(ls $SUP/*.png | wc -l) supervision maps"
S1D=$O/splat_runs_STAGE1/stage1_bg00_glref/high/h3dgs_sidecar; mkdir -p $S1D/nerfstudio_models
(cd $NS && pixi run python /home/paperspace/logs/sidecar_dataparser.py $O $S1D 2>&1 | grep -a '^\[dataparser\]' | tee -a $L)
[ -e $S1D/dataparser_transforms.json ] || { say "chunk $CN: dataparser failed"; exit 1; }
$PYH /home/paperspace/logs/h3dgs_to_stage1.py --block $O --h3dgs $P --hier $HIER --template $TEMPLATE --dataparser $S1D/dataparser_transforms.json \
  --out $S1D/nerfstudio_models/step-000015000.ckpt --margin $MARGIN --cell $P/camera_calibration/chunks/$CN 2>&1 | grep -a '^\[sidecar\]' | cut -c1-260 | tee -a $L
[ -e $S1D/nerfstudio_models/step-000015000.ckpt ] || { say "chunk $CN: conversion failed"; exit 1; }
mkdir -p $O/stage2_init_glref/nerfstudio_models
$PYH - $S1D/nerfstudio_models/step-000015000.ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); k = [k for k in ck['pipeline'] if k.endswith('gauss_params.means')][0]
ck['pipeline'][k.replace('means', 'high_features')] = torch.zeros((ck['pipeline'][k].shape[0], 32)); torch.save(ck, sys.argv[2]); print(f"[init0] {ck['pipeline'][k].shape[0]} gaussians + zero features", flush=True)
PY
cd $NS; t1=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name stage2_bootstrap_glref \
  --load-dir $O/stage2_init_glref/nerfstudio_models --pipeline.model.freeze-geometry True --pipeline.model.high-loss-weight 1.0 --pipeline.datamanager.semantic-dir $SUP \
  --pipeline.model.rasterize-mode antialiased --pipeline.model.sky-loss-lambda 0.0 --max-num-iterations 1 --steps-per-save 19998 --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 \
  > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}_bootstrap.log 2>&1 || true
BOOT=$(ls -t $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/config.yml 2>/dev/null | head -1); [ -n "$BOOT" ] || { say "chunk $CN: no bootstrap config (see sidecar_${SV}_chunk_${CN}_bootstrap.log)"; exit 1; }
say "chunk $CN bootstrap in $(( $(date +%s)-t1 ))s"
t1=$(date +%s); W=$O/splat_runs_FEATFIX/interaction_W_glref_bg.npz
CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$BOOT" --supervision-dir $SUP --out-npz $W > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}_census.log 2>&1
[ -e $W ] || { say "chunk $CN: census failed"; exit 1; }; say "chunk $CN census in $(( $(date +%s)-t1 ))s"
INIT=$O/stage2_init_census_glref_bg_f1.0_r2/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt --dst-dir $INIT --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'background|floors|assigned' | tee -a $L
TS=$(basename $(dirname $BOOT)); RUN=$O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/$TS; mkdir -p $RUN/nerfstudio_models
mv $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_glref_bg_f1.0_r2|" $BOOT > $RUN/config.yml; cp $(dirname $BOOT)/dataparser_transforms.json $RUN/ 2>/dev/null || true
$PYH -c "
import torch,sys,glob; p=glob.glob(sys.argv[1]+'/*.ckpt')[0]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['optimizers']={}; ck['schedulers']={}; torch.save(ck,p)" "$RUN/nerfstudio_models"
rm -rf $O/splat_runs_STAGE1 $O/stage2_init_glref $O/stage2_init_census_glref_bg_f1.0_r2 $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/nerfstudio_models*
FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --survey-root $S --out $O --proj $P 2>>$L | head -1)
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/chunk_${CN}_bg_f1.0_r2_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[chunk_$CN sidecar bg_f1.0_r2 $FR] /" >> $VL
rm -rf $O/clip_cache_*
say "phone chunk $CN side-car ready in $(( ($(date +%s)-t0)/60 )) min; run $RUN; verdict $FR: $(grep -aE "^\[chunk_$CN sidecar" $VL | grep -oE '(TREE|ROW) of [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | head -8 | tr '\n' ';')"
