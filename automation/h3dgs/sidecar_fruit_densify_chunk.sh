#!/bin/bash
# sidecar_fruit_densify_chunk.sh <survey> <side-car dir name, e.g. chunk_0_0_expo> <chunk> [iters=2000] [share=0.1]
# FRUIT DENSIFICATION on a CHUNK side-car (2026-09-30) — the block recipe of sidecar_fruit_densify.sh (Paul, 2026-09-27) run
# on a chunk side-car under experimental/h3dgs_sidecar_chunks: resume the settled ratio-2 seed with features zeroed and
# frozen, fruit-protect + fruit-densify split the coarse carriers of fruit supervision, re-census with the void row,
# reseed (ratio 2 + fruit share-assign), verdict on the top-FRUIT frame whose camera is inside the cell.
# Fruit supervision for the chunk is assembled from the owning blocks' trees_fruit_v3 (sidecar_chunk_fruitsup.py) with a
# merged manifest, which is what the render service reads for fruit words. SIDECAR_PROJ = H3DGS project (cell files).
set -uo pipefail
SV=${1:?survey}; DN=${2:?side-car dir name}; CN=${3:?chunk}; ITERS=${4:-2000}; SHARE=${5:-0.1}
S=/home/paperspace/data/citrus_all/$SV; P=${SIDECAR_PROJ:-$S/experimental/h3dgs}; O=$S/experimental/h3dgs_sidecar_chunks/$DN; SUP=$O/supervision/trees_fruit_v3
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=${SIDECAR_KF:-$S/prod/scratch_sam3}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_fruitdensify.log; VL=/home/paperspace/logs/sidecar_${SV}_fruit_verdicts.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
SEED=$(ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1)
[ -n "$SEED" ] || { say "densify $DN: missing ratio-2 seed"; exit 1; }
EXP=${FD_EXP:-fruit_densify}; RS=${FD_EXP:+_$FD_EXP}
if ls $O/splat_runs_FEATFIX/stage2_censusinit_fruitdensify${RS}_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1; then say "densify $DN ($EXP): already done, skipping"; exit 0; fi
python3 /home/paperspace/logs/sidecar_chunk_fruitsup.py $SV $O 2>&1 | grep -a '^\[chunk-fruitsup\]' | tee -a $L
[ -e $SUP/manifest.json ] && ls $SUP/kf_*.png > /dev/null 2>&1 || { say "densify $DN: no fruit supervision assembled"; exit 1; }
STEP0=$(basename $SEED | grep -oE '[0-9]+' | sed 's/^0*//'); MAXIT=$((STEP0 + ITERS + 1))
# The HiGH gate `fruit_densify_scale` (doc: metres) is compared against exp(scales) in MODEL units (high_model.py:204-206), so
# its world threshold is 0.04 / dataparser_scale (1.13 m on chunk_1_0_expo, 0.40 m on block_018) and the boost never fired on any
# side-car run (0 "[fruit-densify] step=" lines, 2026-10-01). FD_SCALE_M / FD_MAXSCALE_M (metres) are converted to model units
# here through the seed run's dataparser scale; unset = HiGH defaults as before (0.04 / 0.5 model units).
DPS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scale'])" "$(dirname $(dirname $SEED))/dataparser_transforms.json")
FD_ARGS=""
if [ -n "${FD_SCALE_M:-}" ]; then FD_ARGS="--pipeline.model.fruit-densify-scale $(python3 -c "print($FD_SCALE_M*$DPS)") --pipeline.model.fruit-densify-max-scale $(python3 -c "print(${FD_MAXSCALE_M:-0.5}*$DPS)")"; fi
t0=$(date +%s); say "=== fruit densify $DN: resume step $STEP0 -> $MAXIT ($ITERS iters), features frozen at zero, splits on; dataparser scale $DPS; gate ${FD_SCALE_M:-default(0.04 model units = $(python3 -c "print(round(0.04/$DPS,2))") m)}${FD_SCALE_M:+ m -> $FD_ARGS}"
Z=$O/fruitdensify_init/nerfstudio_models; mkdir -p $Z
cd $NS && pixi run python - "$SEED" "$Z/$(basename $SEED)" << 'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
k = [k for k in ck['pipeline'] if k.endswith('gauss_params.high_features')][0]
ck['pipeline'][k] = torch.zeros_like(ck['pipeline'][k]); ck['optimizers'] = {}; ck['schedulers'] = {}
n = [v for kk, v in ck['pipeline'].items() if kk.endswith('gauss_params.means')][0].shape[0]
torch.save(ck, sys.argv[2]); print(f'[densify-init0] {n} gaussians, features zeroed', flush=True)
PY
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB HIGH_LOSS_WARMUP_STEP=1000000000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name $EXP \
    --load-dir $Z --pipeline.model.rasterize-mode antialiased --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.fruit-protect True --pipeline.model.fruit-protect-tau 3.0 --pipeline.model.fruit-anchor-weight 0.0 \
    --pipeline.model.fruit-densify True --pipeline.model.fruit-densify-tail ${FD_TAIL:-2000} --pipeline.model.stop-split-at $MAXIT $FD_ARGS \
    --pipeline.model.sky-loss-lambda 1.0 --pipeline.datamanager.semantic-dir $SUP \
    --max-num-iterations $MAXIT --steps-per-save $((MAXIT - 1)) --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 \
    > /home/paperspace/logs/sidecar_${SV}_${DN}_fruitdensify.log 2>&1 || { say "densify $DN: train FAILED"; exit 1; }
DRUN=$(ls -dt $O/splat_runs_FEATFIX/$EXP/high/*/ | head -1); DCK=$(ls -t $DRUN/nerfstudio_models*/*.ckpt | head -1)
[ -n "$DCK" ] || { say "densify $DN: no ckpt"; exit 1; }
grep -aE '^\[fruit-protect\]|^\[fruit-densify\]' /home/paperspace/logs/sidecar_${SV}_${DN}_fruitdensify.log | tail -2 | tee -a $L
say "densify $DN trained in $(( ($(date +%s)-t0)/60 )) min -> $(basename $DCK)"
t1=$(date +%s); W=$O/splat_runs_FEATFIX/interaction_W_fruitdensify${RS}_bg.npz
CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$DRUN/config.yml" \
  --supervision-dir $SUP --out-npz $W > /home/paperspace/logs/sidecar_${SV}_${DN}_fruitdensify_census.log 2>&1
[ -e $W ] || { say "densify $DN: census failed"; exit 1; }; say "census in $(( $(date +%s)-t1 ))s"
INIT=$O/stage2_init_fruitdensify${RS}_r2/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $DCK --dst-dir $INIT \
  --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign $SHARE 2>&1 | grep -aE 'background|floors|share-assign|assigned' | tee -a $L
RUN=$O/splat_runs_FEATFIX/stage2_censusinit_fruitdensify${RS}_r2/high/$(basename $DRUN); mkdir -p $RUN/nerfstudio_models
mv $INIT/*.ckpt $RUN/nerfstudio_models/; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_fruitdensify${RS}_r2|" $DRUN/config.yml > $RUN/config.yml
cp $DRUN/dataparser_transforms.json $RUN/ 2>/dev/null || true; rm -rf $Z $INIT
FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $P --fruit 2>>$L | head -1)
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG
HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ \
  --supervision-dir $SUP --frame $FR --kf-images $KF --out $FIG/fruitdensify${RS}_${DN}_$FR 2>&1 \
  | grep -aE "^\[containment\] frame|^(TREE|ROW|FRUIT) " | sed "s/^/[$DN sidecar fruitdensify${RS} $FR] /" | tee -a $VL | cut -c1-150
rm -rf $O/clip_cache_*
say "fruit densify $DN done in $(( ($(date +%s)-t0)/60 )) min; run $RUN; fruit verdict: $(grep -aE "^\[$DN sidecar fruitdensify${RS} " $VL | grep -oE 'FRUIT of [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
