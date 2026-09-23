#!/bin/bash
# Calibrated-intrinsics chain (Paul, 2026-09-23: "Stop this undistortion path"): the ORIGINAL frames with the
# target-calibrated focal length and principal point, exported from the chosen block transforms; depth maps
# hardlinked from a sibling project on the same frames when available; then the standard recipe for the chunk(s)
# given, per-chunk scores for a single chunk, merge + survey-level score (+ Tassili serve) for "all".
#   usage: h3dgs_calib_chain.sh <survey_root> <proj_dir> <chunk|all> <transforms name> "fx,fy,cx,cy" [depth donor proj]
S=${1:?survey}; P=${2:?proj}; C=${3:?chunk|all}; TJ=${4:?transforms name}; INTR=${5:?fx,fy,cx,cy}; DONOR=${6:-}
L=/home/paperspace/logs/h3dgs_calib.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
say "=== calib chain $(basename $S) -> $(basename $P) ($C; poses $TJ; intrinsics $INTR)"
[ -e $P/export_meta.json ] || H3DGS_TRANSFORMS_NAME=$TJ H3DGS_INTRINSICS=$INTR $PY /home/paperspace/logs/h3dgs_export.py $S $P 2>&1 | tail -2 | tee -a $L
mkdir -p $P/camera_calibration/rectified
if [ -n "$DONOR" ] && [ ! -e $P/camera_calibration/rectified/depths ] && [ -e $DONOR/camera_calibration/rectified/depths ]; then cp -al $DONOR/camera_calibration/rectified/depths $P/camera_calibration/rectified/depths; say "depth maps hardlinked from $(basename $DONOR)"; fi
if [ "$C" = all ]; then
  /home/paperspace/logs/h3dgs_survey.sh $S $P > /home/paperspace/logs/h3dgs_survey_$(basename $P)_all.out 2>&1
  say "survey rc=$? ($(grep -aE 'held-out eval|chunks complete|FAILED' /home/paperspace/logs/h3dgs_survey_$(basename $P)_all.out | tail -2 | cut -c18-330 | tr '\n' ';'))"
  [ -e $P/output/merged.hier ] && /home/paperspace/logs/tassili_serve_survey.sh $S $P 2>&1 | grep -aE "8001:|8006:|SERVE" | tee -a $L
else
  H3DGS_ONLY_CHUNKS="$C" /home/paperspace/logs/h3dgs_survey.sh $S $P > /home/paperspace/logs/h3dgs_survey_$(basename $P).out 2>&1
  say "survey rc=$? ($(grep -aE 'SfM in|global BA|train chunk|post-opt chunk|FAILED' /home/paperspace/logs/h3dgs_survey_$(basename $P).out | tail -4 | cut -c18-140 | tr '\n' ';'))"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --out output/eval_chunk_$C --save 12 2>&1 | grep -aE "^\[eval\] tau|Error" | cut -c1-260 | tee -a $L
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --train_sample 400 --out output/fit_chunk_$C --save 6 2>&1 | grep -aE "^\[eval\] tau" | sed 's/^/[fit] /' | cut -c1-260 | tee -a $L
fi
say "CALIB CHAIN $(basename $P) $C DONE"
