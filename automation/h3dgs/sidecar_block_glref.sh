#!/bin/bash
# sidecar_block_glref.sh <survey id> <NNN> [margin m=8] — containment SIDE-CAR for one citrus block (Paul, 2026-09-27):
# H3DGS merged-hierarchy leaves in the block's region -> HiGH stage-1 checkpoint in the block's nerfstudio frame
# (h3dgs_to_stage1.py) -> the census-init stage-2 seed recipe of record (censusinit_block_glref.sh: bootstrap with
# geometry frozen, interaction census, census-init features) -> containment_eval on the block's top supervision frame,
# scored exactly like verdict_sweep_glref.sh, printed beside the block-style verdict (verdicts_censusinit_glref.json).
# Lives beside prod: <survey>/experimental/h3dgs_sidecar/block_NNN/ (transforms.json copy, supervision hardlinks).
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; MARGIN=${3:-8}
S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/tassili/blocks_ns/lio_row100/block_$NNN; O=$S/experimental/h3dgs_sidecar/block_$NNN
H3=$S/experimental/h3dgs; EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
L=/home/paperspace/logs/sidecar_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
S1T=$(ls -t $B/splat_runs_STAGE1/stage1_bg00_glref/high/*/nerfstudio_models/*.ckpt | head -1); S1D=$(dirname $(dirname $S1T))
[ -n "$S1T" ] && [ -e $H3/output/merged.hier ] && [ -d $B/supervision/trees_only ] || { say "block $NNN: missing stage-1 template / merged.hier / supervision"; exit 1; }
say "=== side-car $SV block $NNN (margin $MARGIN m; template $(basename $S1D))"
# 1. block dir beside prod: poses + supervision (hardlinks) + the converted stage-1 checkpoint in the chain's expected place
mkdir -p $O/supervision/trees_only; cp $B/transforms.json $O/transforms.json
for f in $B/supervision/trees_only/*; do [ -e $O/supervision/trees_only/$(basename $f) ] || ln $f $O/supervision/trees_only/$(basename $f); done
RUN=$O/splat_runs_STAGE1/stage1_bg00_glref/high/h3dgs_sidecar; mkdir -p $RUN/nerfstudio_models; cp $S1D/dataparser_transforms.json $RUN/
if [ ! -e $RUN/nerfstudio_models/$(basename $S1T) ]; then
  $PY /home/paperspace/logs/h3dgs_to_stage1.py --block $B --h3dgs $H3 --hier $H3/output/merged.hier --template $S1T --dataparser $S1D/dataparser_transforms.json \
    --out $RUN/nerfstudio_models/$(basename $S1T) --margin $MARGIN 2>&1 | grep -a '^\[sidecar\]' | cut -c1-300 | tee -a $L
fi
[ -e $RUN/nerfstudio_models/$(basename $S1T) ] || { say "block $NNN: conversion failed"; exit 1; }
# 2. census-init stage-2 seed (recipe of record for the glref era), features only, geometry frozen
t0=$(date +%s)
CENSUS_EMBEDDER=$EMB CENSUS_HIERARCHY=$HJ bash /home/paperspace/logs/censusinit_block_glref.sh $O > /home/paperspace/logs/sidecar_${SV}_b${NNN}_chain.log 2>&1
say "block $NNN chain rc=$? in $(( $(date +%s)-t0 ))s: $(grep -aE 'REPL-' /home/paperspace/logs/sidecar_${SV}_b${NNN}_chain.log | tail -3 | cut -c1-160 | tr '\n' '|')"
CFG=$(ls -t $O/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml 2>/dev/null | head -1); [ -n "$CFG" ] || { say "block $NNN: no side-car seed"; exit 1; }
# 3. verdict: same frame rule and scorer as verdict_sweep_glref.sh
SUP=$O/supervision/trees_only; cd /home/paperspace/code/nerf_new
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
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; t0=$(date +%s)
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python /home/paperspace/code/aru_sil_core/src/scripts/containment_eval.py \
  --config $CFG --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/block_${NNN}_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[$NNN $FR] /" | tee -a $VL
say "block $NNN verdict on $FR in $(( $(date +%s)-t0 ))s — SIDE-CAR: $(grep -aE "^\[$NNN " $VL | grep -oE '(TREE|ROW|FRUIT)[^"]*"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "block $NNN BLOCK-STYLE reference: $(python3 -c "
import json; d=dict(json.load(open('$S/prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_glref.json'))['blocks']).get('$NNN'); print(d['frame'], 'trees', d['trees'], 'rows', d['rows']) if d else print('none')")"
say "=== side-car block $NNN DONE"
