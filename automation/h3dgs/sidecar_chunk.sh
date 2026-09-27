#!/bin/bash
# sidecar_chunk.sh <survey id> <chunk> [margin m=6] — CHUNK-level containment side-car (Paul, 2026-09-27: "why blocks and
# not chunks, I thought we were using H3DGS"): dataset from the chunk's own poses + supervision union -> nerfstudio
# dataparser frame -> H3DGS leaves in the chunk cell (+ margin) as a stage-1 ckpt -> zero-feature init -> 1-iteration
# bootstrap (geometry frozen; writes the config) -> ONE census with the void row -> ratio-2 seed (recipe settled on
# the 05 blocks) -> seed run dir; transient checkpoints and the CLIP cache removed; verdict on the chunk's top frame.
# Output: <survey>/experimental/h3dgs_sidecar_chunks/chunk_<c>/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/<ts>/
set -uo pipefail
SV=${1:?survey}; CN=${2:?chunk}; MARGIN=${3:-6}
S=/home/paperspace/data/citrus_all/$SV; P=$S/experimental/h3dgs; O=$S/experimental/h3dgs_sidecar_chunks/chunk_$CN; SUP=$O/supervision/trees_only
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_chunks.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
TEMPLATE=$(ls $S/experimental/h3dgs_sidecar/block_020/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt | head -1)
if ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1; then say "chunk $CN: seed exists, skipping"; exit 0; fi
t0=$(date +%s); say "=== chunk $CN side-car (margin $MARGIN m)"
# 1. dataset + dataparser frame
$PYH /home/paperspace/logs/sidecar_chunk_dataset.py $SV $CN $O 2>&1 | grep -a '^\[chunk-dataset\]' | tee -a $L
S1D=$O/splat_runs_STAGE1/stage1_bg00_glref/high/h3dgs_sidecar; mkdir -p $S1D/nerfstudio_models
(cd $NS && pixi run python /home/paperspace/logs/sidecar_dataparser.py $O $S1D 2>&1 | grep -a '^\[dataparser\]' | tee -a $L)
[ -e $S1D/dataparser_transforms.json ] || { say "chunk $CN: dataparser failed"; exit 1; }
# 2. leaves in the cell -> stage-1 ckpt (no optimizer moments), + zero features = stage2_init
$PYH /home/paperspace/logs/h3dgs_to_stage1.py --block $O --h3dgs $P --hier $P/output/merged.hier --template $TEMPLATE --dataparser $S1D/dataparser_transforms.json \
  --out $S1D/nerfstudio_models/step-000015000.ckpt --margin $MARGIN --cell $P/camera_calibration/chunks/$CN 2>&1 | grep -a '^\[sidecar\]' | cut -c1-260 | tee -a $L
[ -e $S1D/nerfstudio_models/step-000015000.ckpt ] || { say "chunk $CN: conversion failed"; exit 1; }
mkdir -p $O/stage2_init_glref/nerfstudio_models
$PYH - $S1D/nerfstudio_models/step-000015000.ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); k = [k for k in ck['pipeline'] if k.endswith('gauss_params.means')][0]
ck['pipeline'][k.replace('means', 'high_features')] = torch.zeros((ck['pipeline'][k].shape[0], 32)); torch.save(ck, sys.argv[2]); print(f"[init0] {ck['pipeline'][k].shape[0]} gaussians + zero features", flush=True)
PY
# 3. bootstrap (config on this geometry; 1 iteration, geometry frozen) — same flags as censusinit_block_glref.sh
cd $NS; t1=$(date +%s)
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name stage2_bootstrap_glref \
  --load-dir $O/stage2_init_glref/nerfstudio_models --pipeline.model.freeze-geometry True --pipeline.model.high-loss-weight 1.0 --pipeline.datamanager.semantic-dir $SUP \
  --pipeline.model.rasterize-mode antialiased --pipeline.model.sky-loss-lambda 1.0 --max-num-iterations 1 --steps-per-save 19998 --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 \
  > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}_bootstrap.log 2>&1 || true
BOOT=$(ls -t $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/config.yml 2>/dev/null | head -1); [ -n "$BOOT" ] || { say "chunk $CN: no bootstrap config"; exit 1; }
say "chunk $CN bootstrap in $(( $(date +%s)-t1 ))s"
# 4. one census with the void row, ratio-2 seed
t1=$(date +%s); W=$O/splat_runs_FEATFIX/interaction_W_glref_bg.npz
CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$BOOT" --supervision-dir $SUP --out-npz $W > /home/paperspace/logs/sidecar_${SV}_chunk_${CN}_census.log 2>&1
[ -e $W ] || { say "chunk $CN: census failed"; exit 1; }; say "chunk $CN census in $(( $(date +%s)-t1 ))s"
INIT=$O/stage2_init_census_glref_bg_f1.0_r2/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt --dst-dir $INIT --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'background|assigned' | tee -a $L
TS=$(basename $(dirname $BOOT)); RUN=$O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/$TS; mkdir -p $RUN/nerfstudio_models
mv $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_glref_bg_f1.0_r2|" $BOOT > $RUN/config.yml; cp $(dirname $BOOT)/dataparser_transforms.json $RUN/ 2>/dev/null || cp $S1D/dataparser_transforms.json $RUN/
$PYH -c "
import torch,sys,glob; p=glob.glob(sys.argv[1]+'/*.ckpt')[0]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['optimizers']={}; ck['schedulers']={}; torch.save(ck,p)" "$RUN/nerfstudio_models"
rm -rf $O/splat_runs_STAGE1 $O/stage2_init_glref $O/stage2_init_census_glref_bg_f1.0_r2 $O/clip_cache_* $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/nerfstudio_models*
# 5. verdict on the chunk's top-painted frame (sanity; block-frame comparisons are scored separately)
FR=$(pixi run python - "$SUP" << 'PY'
import sys, numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16); n = int((a != 65535).sum())
    if n > best[0]: best = (n, f.name)
print(best[1])
PY
)
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/chunk_${CN}_bg_f1.0_r2_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[chunk_$CN sidecar bg_f1.0_r2 $FR] /" >> $VL
say "chunk $CN side-car ready in $(( ($(date +%s)-t0)/60 )) min; kept $(du -sh $O | cut -f1) ($(df --output=avail -BG / | tail -1 | tr -dc 0-9)G free); verdict $FR: $(grep -aE "^\[chunk_$CN sidecar" $VL | grep -oE '(TREE|ROW)[^"]*"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';' | cut -c1-300)"
