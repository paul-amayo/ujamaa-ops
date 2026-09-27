#!/bin/bash
# sidecar_boost_score.sh <survey id> <NNN> <sidecar|blockstyle> <source seed tag> <min opacity> — stage an identity-proxy
# opacity-boosted copy of a staged seed (sidecar_opacity_boost.py) as its own run dir and score the block's top
# supervision frame with containment_eval (lines tagged "<seed tag>_op<min>" in sidecar_<survey>_verdicts.log).
#   e.g. sidecar_boost_score.sh 05_13D_Jackal 000 sidecar glref_bg_f1.0 0.95
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; KIND=${3:?sidecar|blockstyle}; SRC_TAG=${4:?seed tag}; MINOP=${5:-0.95}
S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/tassili/blocks_ns/lio_row100/block_$NNN
if [ $KIND = sidecar ]; then O=$S/experimental/h3dgs_sidecar/block_$NNN; else O=$S/experimental/h3dgs_sidecar/blockstyle_ref/block_$NNN; fi
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3; SUP=$B/supervision/trees_only
ARU=/home/paperspace/code/aru_sil_core/src/scripts; L=/home/paperspace/logs/sidecar_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
SRC=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_${SRC_TAG}/high/* | head -1); [ -n "$SRC" ] || { say "$KIND block $NNN: no seed run stage2_censusinit_${SRC_TAG}"; exit 1; }
TAG=${SRC_TAG}_op${MINOP}; RUN=$O/splat_runs_FEATFIX/stage2_censusinit_${TAG}/high/$(basename $SRC); mkdir -p $RUN/nerfstudio_models
CK=$(ls $SRC/nerfstudio_models/*.ckpt | head -1); cd /home/paperspace/code/nerf_new
pixi run python /home/paperspace/logs/sidecar_opacity_boost.py $CK $RUN/nerfstudio_models/$(basename $CK) --min-opacity $MINOP 2>&1 | grep -a '^\[boost\]' | tee -a $L
sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_${TAG}|" $SRC/config.yml > $RUN/config.yml; cp $SRC/dataparser_transforms.json $RUN/ 2>/dev/null || true
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
CONTAIN_DEBUG=1 HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/${KIND}_block_${NNN}_${TAG}_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) |^\[debug\] label|^\[norm gate\]" | sed "s/^/[$NNN $KIND $TAG $FR] /" | tee -a $VL
say "$KIND block $NNN $TAG verdict on $FR in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[$NNN $KIND $TAG " $VL | grep -oE '(TREE|ROW|FRUIT)[^"]*"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
