#!/bin/bash
# sidecar_bg_reseed.sh <survey id> <NNN> <sidecar|blockstyle> [tree floor=1.0] — re-census a block's stage-2 seed WITH the
# unlabelled-pixel row (CENSUS_WITH_BG=1) and re-seed with --bg-competes (gaussians whose void mass beats their best
# label get no identity), then score the top supervision frame with containment_eval. `sidecar` works on the H3DGS
# side-car block dir; `blockstyle` builds the same seed for the prod block's own geometry, staged under
# experimental/h3dgs_sidecar/blockstyle_ref/block_NNN (prod untouched) — the like-for-like reference for the rule.
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; KIND=${3:?sidecar|blockstyle}; FLOOR=${4:-1.0}
S=/home/paperspace/data/citrus_all/$SV; TAG=glref; B=$S/prod/tassili/blocks_ns/lio_row100/block_$NNN
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; L=/home/paperspace/logs/sidecar_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
if [ $KIND = sidecar ]; then O=$S/experimental/h3dgs_sidecar/block_$NNN; else O=$S/experimental/h3dgs_sidecar/blockstyle_ref/block_$NNN; fi
SRC=$O   # blockstyle: the prod block's own stage-1 ckpt chained under experimental first (prod's bootstrap / stage2_init were reclaimed)
BOOT=$(ls -t $SRC/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml | head -1); S1=$(ls $SRC/stage2_init_${TAG}/nerfstudio_models/*.ckpt | head -1); SUP=$B/supervision/trees_only
[ -n "$BOOT" ] && [ -n "$S1" ] || { say "$KIND block $NNN: no bootstrap config / stage2_init"; exit 1; }
mkdir -p $O/splat_runs_FEATFIX; W=$O/splat_runs_FEATFIX/interaction_W_${TAG}_bg.npz; TS=$(basename $(dirname $BOOT)); FT=bg_f${FLOOR}
INIT=$O/stage2_init_census_${TAG}_${FT}/nerfstudio_models; RUN=$O/splat_runs_FEATFIX/stage2_censusinit_${TAG}_${FT}/high/$TS
cd /home/paperspace/code/nerf_new; say "--- $KIND block $NNN: census with the unlabelled row, seed with --bg-competes, tree floor $FLOOR"
t0=$(date +%s)
[ -e $W ] || CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$BOOT" --supervision-dir $SUP --out-npz $W 2>&1 | grep -aE 'saved W|Error|error' | tail -2 | tee -a $L
[ -e $W ] || { say "$KIND block $NNN: census failed"; exit 1; }
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $S1 --dst-dir $INIT --tree-floor $FLOOR --bg-competes 2>&1 | grep -aE 'background|floors|assigned' | tee -a $L
mkdir -p $RUN/nerfstudio_models; cp $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_${TAG}_${FT}|" $BOOT > $RUN/config.yml
cp $(dirname $BOOT)/dataparser_transforms.json $RUN/ 2>/dev/null || cp $(ls -t $SRC/splat_runs_STAGE1/stage1_bg00_${TAG}/high/*/dataparser_transforms.json | head -1) $RUN/
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
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log
CONTAIN_DEBUG=1 HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
  --out $FIG/${KIND}_block_${NNN}_${FT}_$FR 2>&1 | grep -aE "^(TREE|ROW|FRUIT) |^\[debug\] unlabelled|^\[norm gate\]" | sed "s/^/[$NNN $KIND $FT $FR] /" | tee -a $VL
say "$KIND block $NNN $FT verdict on $FR in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[$NNN $KIND $FT " $VL | grep -oE '(TREE|ROW|FRUIT)[^"]*"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
