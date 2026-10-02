#!/bin/bash
# stage2_train_chunk.sh — LIVE stage-2 refine from a census seed on an H3DGS chunk side-car (2026-10-02, Paul: "remove the
# sidecar option and have two stage h3dgs census training like we did for blocks").
# Record: the block fleet was seed-only (CENSUS_SEED_ONLY=1) and the one fw2 refine that justified it moved features by
# 0.00000 (dead optimizer, 2026-08) — so a stage 2 with a live optimizer has never been measured on any cell. This runs it:
# the seed ckpt carries NO optimizer/scheduler state (fresh Adam at lr 1e-3 for high_features), geometry frozen, fw2 flags,
# saves every SAVE iters, then scores containment on the given frames and the per-gaussian feature displacement vs the seed.
#   env: O (side-car dir) SEED (seed ckpt) SUP (supervision dir) EMB HJ KF TAG ITERS [SAVE=1000] [FRAMES="a.png b.png"]
#        [FRUIT=0|1 -> --fruit-protect + fruit weight 2] [SV=survey name for the verdict log] [SCORE_MID=1 -> also score the first save]
set -uo pipefail
: "${O:?} ${SEED:?} ${SUP:?} ${EMB:?} ${HJ:?} ${KF:?} ${TAG:?} ${ITERS:?}"; SAVE=${SAVE:-1000}; FRAMES=${FRAMES:-}; FRUIT=${FRUIT:-0}; SV=${SV:-chunk}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/stage2_train_chunk.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
STEP0=$(basename $SEED | grep -oE '[0-9]+' | sed 's/^0*//'); MAXIT=$((STEP0 + ITERS + 1)); EXP=stage2_train_$TAG
INIT=$O/stage2_train_init_$TAG/nerfstudio_models; mkdir -p $INIT; ln -sfn $SEED $INIT/$(basename $SEED)
FR_ARGS=""; [ "$FRUIT" = "1" ] && FR_ARGS="--pipeline.model.fruit-protect True --pipeline.model.high-loss-fruit-weight 2.0"
say "=== stage2 live refine $TAG: seed $(basename $SEED) step $STEP0 -> $MAXIT ($ITERS iters, save every $SAVE), geometry frozen, fresh optimizer; sup $(basename $SUP); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
t0=$(date +%s); cd $NS
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  pixi run ns-train high --data $O --output-dir $O/splat_runs_FEATFIX --experiment-name $EXP --load-dir $INIT \
    --pipeline.model.freeze-geometry True --pipeline.model.high-loss-weight 1.0 $FR_ARGS \
    --pipeline.datamanager.semantic-dir $SUP --pipeline.model.rasterize-mode antialiased --pipeline.model.sky-loss-lambda 1.0 \
    --max-num-iterations $MAXIT --steps-per-save $SAVE --save-only-latest-checkpoint False \
    --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 > /home/paperspace/logs/stage2_train_${SV}_$TAG.log 2>&1 || { say "$TAG: train FAILED (rc=$?) — see stage2_train_${SV}_$TAG.log"; exit 1; }
RUN=$(ls -dt $O/splat_runs_FEATFIX/$EXP/high/*/ | head -1); tt=$(( $(date +%s)-t0 ))
CKS=$(ls $RUN/nerfstudio_models/*.ckpt | sort); LAST=$(echo "$CKS" | tail -1)
say "$TAG trained in $((tt/60)) min ($(python3 -c "print(round($ITERS/max($tt,1),2))") it/s) -> $(echo "$CKS" | xargs -n1 basename | paste -sd' ')"
tr '\r' '\n' < /home/paperspace/logs/stage2_train_${SV}_$TAG.log | grep -aoE "high_loss[^0-9]*[0-9.e+-]+" | awk 'NR==1||NR%200==0{print}' | tail -8 | paste -sd' | ' | sed 's/^/[loss] /' | tee -a $L
# feature displacement vs the seed (the metric that read 0.00000 in August)
pixi run python - "$SEED" "$LAST" <<'PY' 2>&1 | tee -a $L
import sys, torch
a=torch.load(sys.argv[1],map_location='cpu',weights_only=False); b=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
fa=[v for k,v in a['pipeline'].items() if k.endswith('gauss_params.high_features')][0].float(); fb=[v for k,v in b['pipeline'].items() if k.endswith('gauss_params.high_features')][0].float()
seeded=fa.abs().sum(1)>0; d=(fb-fa).norm(dim=1); nb=fb.norm(dim=1)
print(f"[move] seeded rows {int(seeded.sum()):,}/{fa.shape[0]:,}: median displacement {d[seeded].median():.4f} (seed median norm {fa[seeded].norm(dim=1).median():.3f}); unseeded rows now with norm>0.5: {int((nb[~seeded]>0.5).sum()):,} ({100*float((nb[~seeded]>0.5).float().mean()):.1f}%), median norm of those {nb[~seeded][nb[~seeded]>0.5].median() if (nb[~seeded]>0.5).any() else 0:.3f}; rows moved >0.01: {int((d>0.01).sum()):,}")
PY
score(){ # $1 ckpt, $2 tag, frames...
  local CK=$1 T=$2; shift 2; local D=$O/splat_runs_FEATFIX/${EXP}_eval_$T/high/$(basename $RUN); mkdir -p $D/nerfstudio_models; ln -sfn $CK $D/nerfstudio_models/$(basename $CK)
  sed "s|^experiment_name: .*$|experiment_name: ${EXP}_eval_$T|" $RUN/config.yml > $D/config.yml; cp $RUN/dataparser_transforms.json $D/ 2>/dev/null || true
  for FR in "$@"; do
    HIGH_EMBEDDER_CKPT=$EMB timeout 900 pixi run python $ARU/containment_eval.py --config $D/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF --out /home/paperspace/logs/sidecar_figs_$SV/${EXP}_${T}_$FR > /home/paperspace/logs/sidecar_${SV}_${EXP}_${T}_verdict_$FR.log 2>&1
    grep -aE '^(TREE|ROW|FRUIT) ' /home/paperspace/logs/sidecar_${SV}_${EXP}_${T}_verdict_$FR.log | sed "s/^/[$EXP $T $FR] /" >> $VL
    say "verdict $T $FR: $(grep -aE "^\[$EXP $T $FR\]" $VL | sed -E 's/ +/ /g' | grep -oE '(TREE|ROW|FRUIT of) [0-9]+ "[a-z]+": thr [0-9.na]+ IoU [0-9.]+' | tr '\n' ';')$(grep -aE 'Error|no supervision frame' /home/paperspace/logs/sidecar_${SV}_${EXP}_${T}_verdict_$FR.log | head -1)"; rm -rf $O/clip_cache_*
  done
}
if [ -n "$FRAMES" ]; then
  score $LAST step$(( $(basename $LAST | grep -oE '[0-9]+' | sed 's/^0*//') - STEP0 )) $FRAMES
  if [ "${SCORE_MID:-0}" = "1" ]; then MID=$(echo "$CKS" | head -1); [ "$MID" != "$LAST" ] && score $MID step$(( $(basename $MID | grep -oE '[0-9]+' | sed 's/^0*//') - STEP0 )) $(echo $FRAMES | cut -d' ' -f1); fi
fi
rm -rf $O/stage2_train_init_$TAG
say "=== stage2 live refine $TAG done in $(( ($(date +%s)-t0)/60 )) min -> $RUN"
