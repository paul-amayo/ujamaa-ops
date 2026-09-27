#!/bin/bash
# sidecar_reinit.sh <survey id> <NNN> <tree floor> — re-seed a side-car block's features from its existing interaction
# census with a different assignment floor (build_census_init.py --tree-floor; recipe of record 1.0), stage the seed as
# a run dir beside the others (stage2_censusinit_glref_f<floor>/high/<bootstrap ts>/) and score the SAME top supervision
# frame with containment_eval, appending floor-tagged verdict lines to sidecar_<survey>_verdicts.log.
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; FLOOR=${3:?tree floor}
S=/home/paperspace/data/citrus_all/$SV; O=$S/experimental/h3dgs_sidecar/block_$NNN; TAG=glref
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
L=/home/paperspace/logs/sidecar_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
ARU=/home/paperspace/code/aru_sil_core/src/scripts; W=$O/splat_runs_FEATFIX/interaction_W_${TAG}.npz; S1=$(ls $O/stage2_init_${TAG}/nerfstudio_models/*.ckpt | head -1)
BOOT=$(ls -t $O/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml | head -1); TS=$(basename $(dirname $BOOT))
[ -e $W ] && [ -n "$S1" ] && [ -n "$BOOT" ] || { say "block $NNN: no census / init / bootstrap to re-seed from"; exit 1; }
FT=f${FLOOR}; INIT=$O/stage2_init_census_${TAG}_${FT}/nerfstudio_models; RUN=$O/splat_runs_FEATFIX/stage2_censusinit_${TAG}_${FT}/high/$TS
cd /home/paperspace/code/nerf_new
say "--- block $NNN re-seed with tree floor $FLOOR"
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $S1 --dst-dir $INIT --tree-floor $FLOOR 2>&1 | grep -aE 'floors|assigned' | tee -a $L
mkdir -p $RUN/nerfstudio_models; cp $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_${TAG}_${FT}|" $BOOT > $RUN/config.yml; cp $(dirname $BOOT)/dataparser_transforms.json $RUN/ 2>/dev/null || cp $O/splat_runs_STAGE1/stage1_bg00_${TAG}/high/*/dataparser_transforms.json $RUN/
SUP=$O/supervision/trees_only
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
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/block_${NNN}_${FT}_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[$NNN $FT $FR] /" | tee -a $VL
say "block $NNN floor $FLOOR verdict on $FR in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[$NNN $FT " $VL | grep -oE '(TREE|ROW|FRUIT)[^"]*"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
