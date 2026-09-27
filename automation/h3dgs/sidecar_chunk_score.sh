#!/bin/bash
# sidecar_chunk_score.sh <survey> <chunk>... — re-score an already-built chunk side-car on the top-supervised frame whose
# camera is INSIDE the chunk cell (sidecar_chunk_frame.py). Scoring only; nothing is rebuilt.
set -uo pipefail
SV=${1:?survey}; shift
S=/home/paperspace/data/citrus_all/$SV; EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3; ARU=/home/paperspace/code/aru_sil_core/src/scripts
L=/home/paperspace/logs/sidecar_${SV}_chunks.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; FIG=/home/paperspace/logs/sidecar_figs_$SV
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
cd /home/paperspace/code/nerf_new
for CN in "$@"; do
  O=$S/experimental/h3dgs_sidecar_chunks/chunk_$CN; SUP=$O/supervision/trees_only
  RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ 2>/dev/null | head -1)
  [ -n "$RUN" ] || { say "rescore $CN: no seed run"; continue; }
  FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame.py $SV $CN 2>>$L | head -1)
  [ -n "$FR" ] || { say "rescore $CN: no in-cell supervised frame"; continue; }
  say "rescore chunk $CN on in-cell frame $FR"
  HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ \
    --supervision-dir $SUP --frame $FR --kf-images $KF --out $FIG/chunk_${CN}_incell_$FR 2>&1 | grep -aE "^\[containment\] frame|^(TREE|ROW) " \
    | sed "s/^/[chunk_$CN sidecar incell $FR] /" | tee -a $VL | sed -E 's/ +/ /g' | cut -c1-120
  rm -rf $O/clip_cache_*
done
