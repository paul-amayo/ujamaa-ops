#!/bin/bash
# Per-block pose refinement with ONE FIXED camera for every block (the citrus recipe let GLOMAP move the intrinsics
# per block — on Klapmuts each block ended with a different camera, so neighbouring blocks' poses disagree inside
# every H3DGS chunk). GPU SIFT + exhaustive match + GLOMAP with --BundleAdjustment.optimize_intrinsics 0, the
# PINHOLE camera taken from the odometry transforms (= what the export and H3DGS use), then Sim(3) into the LIO
# world against the odometry poses, written to a NAMED transforms file (transforms.json untouched).
#   usage: klapmuts_refine_fixedcam.sh <survey_root> [out name = transforms_ref_fixedcam.json]
SV=${1:?survey root}; OUT=${2:-transforms_ref_fixedcam.json}; CFG=$SV/prod/tassili/blocks_ns/lio_row100
LOG=/home/paperspace/logs/klapmuts_refine_fixedcam_$(basename $SV).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap; GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
say "=== fixed-camera refine $(basename $SV) -> $OUT: $(ls -d $CFG/block_[0-9][0-9][0-9] | wc -l) canonical blocks"
for BD in $CFG/block_[0-9][0-9][0-9]; do
  B=$(basename $BD); t0=$(date +%s); BASE=transforms_odo_glfix.json; [ -e $BD/$BASE ] || BASE=transforms.json
  [ -e $BD/$OUT ] && { say "$B already done — skip"; continue; }
  W=$BD/colmap_glref_fixedcam; rm -rf $W; mkdir -p $W/images $W/sparse
  PAR=$(python3 -c "
import json; t = json.load(open('$BD/$BASE')); print('%.6f,%.6f,%.6f,%.6f' % (t['fl_x'], t['fl_y'], t['cx'], t['cy']))")
  python3 -c "
import json, os
from pathlib import Path
t = json.load(open('$BD/$BASE'))
for f in t['frames']:
    src = Path(f['file_path']); dst = Path('$W/images')/src.name
    if not dst.exists(): os.link(src, dst)
print('[refine] staged', len(t['frames']), 'kf images, camera $PAR (fixed)')" >> $LOG 2>&1
  export LD_LIBRARY_PATH=$GLD
  $CC feature_extractor --database_path $W/database.db --image_path $W/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PAR" --FeatureExtraction.use_gpu 1 > $W/sfm.log 2>&1 || { say "$B FEAT FAILED"; continue; }
  $CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $W/sfm.log 2>&1 || { say "$B MATCH FAILED"; continue; }
  $GB mapper --database_path $W/database.db --image_path $W/images --output_path $W/sparse --BundleAdjustment.optimize_intrinsics 0 --BundleAdjustment.optimize_principal_point 0 >> $W/sfm.log 2>&1 || { say "$B GLOMAP FAILED"; continue; }
  unset LD_LIBRARY_PATH
  [ -d $W/sparse/0 ] || { say "$B: no model"; continue; }
  python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
  python3 /home/paperspace/logs/klapmuts_apply_refine_named.py $BD $W $BASE $OUT 2>&1 | tee -a $LOG | tail -1 | sed "s/^/[$(date '+%m-%d %H:%M:%S')] /"
  rm -rf $W/images $W/database.db*   # transient (hardlinks + features); the sparse model stays for the record
  say "$B done in $(( $(date +%s)-t0 ))s"
done
say "FIXEDCAM REFINE DONE: $(ls $CFG/block_[0-9][0-9][0-9]/$OUT 2>/dev/null | wc -l) blocks written"
