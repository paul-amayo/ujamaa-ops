#!/bin/bash
# Whole-clip cabbage, stage A on the CPU (Paul 2026-10-04: "re-segment and reconstruct the whole cabbage video"; the GPU is
# held by the 05 retrain's post-opt). SIFT + exhaustive matching over all 126 frames of IMG_7993 (the original database was
# reclaimed) -> the consecutive-inlier profile that cut frames 0-53 as a "turn". Same COLMAP binary and camera model as
# image_farm_prep.py, GPU flags off.
set -uo pipefail
C=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993; DB=$C/database_gpu.db; L=/home/paperspace/logs/cabbage_whole_clip.log
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap
export LD_LIBRARY_PATH=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -s $DB ] || { say "SIFT (CPU) on $(ls $C/images | wc -l) frames"; $CC feature_extractor --database_path $DB --image_path $C/images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 0 >> $L 2>&1
  say "exhaustive matching (CPU)"; $CC exhaustive_matcher --database_path $DB --FeatureMatching.use_gpu 0 >> $L 2>&1; }
/home/paperspace/miniconda3/envs/h3dgs/bin/python - $DB <<'PY' | tee -a $L
import sqlite3, re, sys, numpy as np
con = sqlite3.connect(sys.argv[1]); num = {i: int(re.search(r'(\d+)', n).group(1)) for i, n in con.execute('select image_id, name from images')}
inl = {}
for pid, rows in con.execute('select pair_id, rows from two_view_geometries'):
    a, b = pid // 2147483647, pid % 2147483647; inl[(num[a], num[b])] = rows; inl[(num[b], num[a])] = rows
n = max(num.values()) + 1; c = np.array([inl.get((k, k + 1), 0) for k in range(n - 1)]); med = float(np.median(c)); thr = max(300.0, 0.35 * med)
print(f'[stageA] {n} frames; consecutive inliers median {med:.0f}, prep threshold {thr:.0f}; pairs below it: {[int(k) for k in np.nonzero(c < thr)[0]]}')
print('[stageA] frames 0-60:', ' '.join(f'{k}:{v}' for k, v in enumerate(c[:61])))
for k in (0, 10, 20, 30, 40, 50, 53): print(f'[stageA] frame {k}: best match into frames 54-125 = {max((inl.get((k, j), 0), j) for j in range(54, n))}')
PY
say "stage A done"
