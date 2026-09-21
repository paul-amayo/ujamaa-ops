#!/bin/bash
# Per-block image-pose refinement for citrus row blocks: GPU SIFT + exhaustive match +
# GLOMAP global mapper on the block's kf images (intrinsics from the block's transforms.json,
# fixed), then Sim(3)-align into the block's LIO world in-place (backup kept).
# usage: citrus_refine_block.sh <block_dir>
set -u
BD=$1; W=$BD/colmap_glref
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap
GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
LOG=$BD/colmap_glref_sfm.log
PAR=$(python3 -c "
import json; t = json.load(open('$BD/transforms.json'))
print('%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%s' % (t['fl_x'], t['fl_y'], t['cx'], t['cy'], t.get('k1',0), t.get('k2',0), t.get('p1',0), t.get('p2',0)))")
mkdir -p $W/images $W/sparse
python3 -c "
import json, shutil
from pathlib import Path
t = json.load(open('$BD/transforms.json'))
for f in t['frames']:
    src = Path(f['file_path']); dst = Path('$W/images')/src.name
    if not dst.exists(): shutil.copy(src, dst)
print('[refine] staged', len(t['frames']), 'kf images')"
export LD_LIBRARY_PATH=$GLD; t0=$(date +%s)
$CC feature_extractor --database_path $W/database.db --image_path $W/images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --ImageReader.camera_params "$PAR" \
  --FeatureExtraction.use_gpu 1 > $LOG 2>&1 || { echo "[refine] FEAT FAILED $BD"; exit 1; }
$CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $LOG 2>&1 || { echo "[refine] MATCH FAILED $BD"; exit 1; }
$GB mapper --database_path $W/database.db --image_path $W/images --output_path $W/sparse >> $LOG 2>&1 || { echo "[refine] GLOMAP FAILED $BD"; exit 1; }
unset LD_LIBRARY_PATH
[ -d $W/sparse/0 ] || { echo "[refine] no model $BD"; exit 1; }
python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
echo "[refine] SfM $(( $(date +%s)-t0 ))s"
python3 /home/paperspace/logs/citrus_apply_refine.py $BD
