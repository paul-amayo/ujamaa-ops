#!/bin/bash
# sidecar_fruit_densify.sh <survey> <NNN> [iters=2000] [share=0.1] — FRUIT DENSIFICATION on a side-car (Paul, 2026-09-27:
# "next step is fruit densification").
#
# WHY: measured this afternoon, the side-car's fruit failure is geometric, not a seeding rule. Block 021's side-car has
# 3.2 M gaussians but only 347 with ANY fruit census mass (134 units total) and 23 above the 0.1 share floor -> IoU 0.002
# against block-style 0.578. The block-style chain gets its fruit from a DENSIFICATION pass (automation/densify_block.sh,
# Paul's 2026-08-26 design): every gaussian big enough to render a fruit also renders thousands of tree pixels, so the
# fruit never owns its own blend until the coarse carriers are SPLIT. The side-car never ran that pass because its
# gaussians are frozen H3DGS leaves.
#
# WHAT THIS DOES: runs that same pass on side-car geometry. Resumes from the settled ratio-2 seed with its features
# zeroed (geometry untouched, features never train: HIGH_LOSS_WARMUP_STEP=1e9 + fruit-anchor-weight 0), at the resumed
# schedule's tail LR so nothing drifts; fruit-protect tallies the gaussians carrying fruit supervision mass and
# fruit-densify boosts the coarse (>4 cm) protected ones past the grow threshold so gsplat splits them, self-limiting at
# the scale gate. Then re-census on the split geometry and reseed. The only intended change is split topology: the
# side-car stops being exactly the H3DGS leaves and becomes the leaves plus fruit children.
# Scored against the block's existing block-style fruit verdict, the same frame rule and scorer as fruit_chain.sh.
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; ITERS=${3:-2000}; SHARE=${4:-0.1}
S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/tassili/blocks_ns/lio_row100/block_$NNN
O=$S/experimental/h3dgs_sidecar/block_$NNN; SUP=$B/supervision/trees_fruit_v3
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_fruitdensify.log; VL=/home/paperspace/logs/sidecar_${SV}_fruit_verdicts.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
SEED=$(ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1)
[ -n "$SEED" ] && [ -d $SUP ] || { say "densify $NNN: missing ratio-2 seed or fruit supervision"; exit 1; }
STEP0=$(basename $SEED | grep -oE '[0-9]+' | sed 's/^0*//'); MAXIT=$((STEP0 + ITERS + 1))
t0=$(date +%s); say "=== fruit densify block $NNN: resume step $STEP0 -> $MAXIT ($ITERS iters), features frozen at zero, splits on"
# 1. zero-feature geometry (the seed's gaussians; features are irrelevant to the split topology and must not train)
Z=$O/fruitdensify_init/nerfstudio_models; mkdir -p $Z
cd $NS && pixi run python - "$SEED" "$Z/$(basename $SEED)" << 'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
k = [k for k in ck['pipeline'] if k.endswith('gauss_params.high_features')][0]
ck['pipeline'][k] = torch.zeros_like(ck['pipeline'][k]); ck['optimizers'] = {}; ck['schedulers'] = {}
n = [v for kk, v in ck['pipeline'].items() if kk.endswith('gauss_params.means')][0].shape[0]
torch.save(ck, sys.argv[2]); print(f'[densify-init0] {n} gaussians, features zeroed', flush=True)
PY
grep -a '\[densify-init0\]' /dev/null 2>/dev/null; say "init0 written"
# 2. the densification pass itself — same flags as automation/densify_block.sh
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB HIGH_LOSS_WARMUP_STEP=1000000000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name fruit_densify \
    --load-dir $Z --pipeline.model.rasterize-mode antialiased --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.fruit-protect True --pipeline.model.fruit-protect-tau 3.0 --pipeline.model.fruit-anchor-weight 0.0 \
    --pipeline.model.fruit-densify True --pipeline.model.fruit-densify-tail ${FD_TAIL:-2000} --pipeline.model.stop-split-at $MAXIT \
    --pipeline.model.sky-loss-lambda 1.0 --pipeline.datamanager.semantic-dir $SUP \
    --max-num-iterations $MAXIT --steps-per-save $((MAXIT - 1)) --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 \
    > /home/paperspace/logs/sidecar_${SV}_b${NNN}_fruitdensify.log 2>&1 || { say "densify $NNN: train FAILED"; exit 1; }
DRUN=$(ls -dt $O/splat_runs_FEATFIX/fruit_densify/high/*/ | head -1); DCK=$(ls -t $DRUN/nerfstudio_models*/*.ckpt | head -1)
[ -n "$DCK" ] || { say "densify $NNN: no ckpt"; exit 1; }
grep -aE '^\[fruit-protect\]|^\[fruit-densify\]' /home/paperspace/logs/sidecar_${SV}_b${NNN}_fruitdensify.log | tail -2 | tee -a $L
say "densify $NNN trained in $(( ($(date +%s)-t0)/60 )) min -> $(basename $DCK); gaussians $(cd $NS && pixi run python -c "
import torch,sys; ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
print([v for k,v in ck['pipeline'].items() if k.endswith('gauss_params.means')][0].shape[0])" "$DCK" 2>/dev/null | tail -1)"
# 3. re-census on the SPLIT geometry, with the void row (the densify run's config already points at the fruit supervision)
t1=$(date +%s); W=$O/splat_runs_FEATFIX/interaction_W_fruitdensify_bg.npz
CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$DRUN/config.yml" \
  --supervision-dir $SUP --out-npz $W > /home/paperspace/logs/sidecar_${SV}_b${NNN}_fruitdensify_census.log 2>&1
[ -e $W ] || { say "densify $NNN: census failed"; exit 1; }; say "census in $(( $(date +%s)-t1 ))s"
# 4. reseed: settled tree rule (ratio 2) + the fruit share-assign, on the densified geometry
INIT=$O/stage2_init_fruitdensify_r2/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $DCK --dst-dir $INIT \
  --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign $SHARE 2>&1 | grep -aE 'background|floors|share-assign|assigned' | tee -a $L
RUN=$O/splat_runs_FEATFIX/stage2_censusinit_fruitdensify_r2/high/$(basename $DRUN); mkdir -p $RUN/nerfstudio_models
mv $INIT/*.ckpt $RUN/nerfstudio_models/; sed 's|^experiment_name: .*$|experiment_name: stage2_censusinit_fruitdensify_r2|' $DRUN/config.yml > $RUN/config.yml
cp $DRUN/dataparser_transforms.json $RUN/ 2>/dev/null || true; rm -rf $Z $INIT $O/fruitdensify_init
# 5. verdict on the block's top-fruit frame, as fruit_chain.sh
FR=$(pixi run python - "$SUP" << 'PY'
import sys, numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16); n = int(((a >= 10000) & (a != 65535)).sum())
    if n > best[0]: best = (n, f.name)
print(best[1])
PY
)
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ \
  --supervision-dir $SUP --frame $FR --kf-images $KF --out $FIG/fruitdensify_${NNN}_$FR 2>&1 \
  | grep -aE "^\[containment\] frame|^(TREE|ROW|FRUIT) " | sed "s/^/[$NNN sidecar fruitdensify $FR] /" | tee -a $VL | cut -c1-150
rm -rf $O/clip_cache_*
say "fruit densify block $NNN done in $(( ($(date +%s)-t0)/60 )) min"
say "  SIDE-CAR densified: $(grep -aE "^\[$NNN sidecar fruitdensify " $VL | grep -oE 'FRUIT of [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "  SIDE-CAR seed-only: $(grep -aE "^\[$NNN sidecar fruit_r2 " $VL | grep -oE 'FRUIT of [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "  BLOCK-STYLE:        $(grep -aE '^FRUIT of' /home/paperspace/logs/fruit_${SV}_b${NNN}_verdict.log 2>/dev/null | sed -E 's/ +/ /g' | cut -c1-70 | tr '\n' ';')"
