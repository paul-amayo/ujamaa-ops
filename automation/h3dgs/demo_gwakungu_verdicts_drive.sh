#!/bin/bash
# demo_gwakungu_verdicts_drive.sh <embedder experiment name> <frame.png ...> — containment verdicts on frames that carry the
# heads the reel's drive actually sees (the three stock verdict frames image_33/40/52 never contained heads 23/24/30).
# Same reseeded run + same tag as demo_gwakungu_final.sh; appends to the survey verdict log the script's honesty rule reads.
set -uo pipefail
EMBN=${1:?embedder experiment name}; shift; FRAMES="$*"
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
C=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage; S=$C/demo_root; O=$S/experimental/h3dgs_sidecar_chunks/chunk_lane; SUP=$O/supervision/trees_only; HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
SV=IMG_7993_s0_cabbage; L=/home/paperspace/logs/demo_chunks_run.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
EMB=$S/prod/bateleur/embedder/$EMBN/ckpts/model_best.pth; TAG=glref_bg_f1.0_r2_$EMBN; RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_$TAG/high/*/ | tail -1)
say "=== gwakungu drive-head verdicts ($EMBN) on: $FRAMES"; t0=$(date +%s); cd $NS
for FR in $FRAMES; do
  HIGH_EMBEDDER_CKPT=$EMB timeout 500 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF --out /home/paperspace/logs/sidecar_figs_$SV/chunk_lane_${TAG}_$FR > /home/paperspace/logs/sidecar_${SV}_chunk_lane_verdict_${TAG}_$FR.log 2>&1
  grep -aE '^(TREE|ROW) ' /home/paperspace/logs/sidecar_${SV}_chunk_lane_verdict_${TAG}_$FR.log | sed "s/^/[chunk_lane sidecar $TAG $FR] /" >> $VL
  say "verdict $FR: $(grep -aE "^\[chunk_lane sidecar $TAG $FR\]" $VL | sed -E 's/ +/ /g' | grep -oE '(TREE|ROW) [0-9]+ "[a-z]+": thr [0-9.na]+ IoU [0-9.]+' | tr '\n' ';')"; rm -rf $O/clip_cache_*
done
say "=== gwakungu drive-head verdicts ($EMBN) done in $(( ($(date +%s)-t0)/60 )) min"
