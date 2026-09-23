#!/bin/bash
# Corrected-camera chain for a Klapmuts survey: wait for the pose-fixed OPENCV camera estimate, re-rectify the
# keyframes with it, export a project from the ODOMETRY poses (LiDAR-derived, camera-independent), run the
# standard recipe (depth -> fixed-pose SfM -> global BA -> chunks -> scaffold -> chunk under test) and score it.
#   usage: h3dgs_camfix_chain.sh <survey_root> <probe_out_file> <proj_dir> <chunk> [transforms name, default transforms_odo_glfix.json]
S=${1:?survey}; PROBE=${2:?probe .out}; P=${3:?proj}; C=${4:?chunk}; TJ=${5:-transforms_odo_glfix.json}
L=/home/paperspace/logs/h3dgs_camfix.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
until grep -q "pose-fixed refined intrinsics" $PROBE 2>/dev/null; do sleep 60; done
CAM=$(grep -a "pose-fixed refined intrinsics" $PROBE | tail -1 | grep -oE "\[[-0-9., e]+\]" | head -1 | tr -d '[] ')
say "=== camfix chain $(basename $S) -> $P (chunk $C, poses from $TJ): OPENCV camera [$CAM]"
[ "$(echo $CAM | tr ',' '\n' | wc -l)" -eq 8 ] || { say "camera parse failed: '$CAM'"; exit 1; }
H3DGS_TRANSFORMS_NAME=$TJ /home/paperspace/logs/h3dgs_undistort_export.sh $S $P "$CAM" || { say "undistort/export FAILED"; exit 1; }
H3DGS_ONLY_CHUNKS="$C" /home/paperspace/logs/h3dgs_survey.sh $S $P > /home/paperspace/logs/h3dgs_survey_$(basename $P).out 2>&1
say "survey rc=$? ($(grep -aE 'global BA|SfM in|train chunk|post-opt chunk|FAILED' /home/paperspace/logs/h3dgs_survey_$(basename $P).out | tail -4 | cut -c18-120 | tr '\n' ';'))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --out output/eval_chunk_$C --save 12 2>&1 | grep -aE "^\[eval\] tau|Error" | cut -c1-260 | tee -a $L
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --train_sample 400 --out output/fit_chunk_$C --save 6 2>&1 | grep -aE "^\[eval\] tau" | sed 's/^/[fit] /' | cut -c1-260 | tee -a $L
say "CAMFIX CHAIN $(basename $P) DONE"
