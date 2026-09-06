#!/bin/bash
# SfM-only axis check for one block: COLMAP poses vs ours -> per-camera-axis dots after Sim(3).
set -uo pipefail; BLK=$1
T=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili; B=$T/blocks_ns/lio_row100; W=$T/colmap_$BLK
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
IPL=/home/paperspace/code/aru_sil_core/src/scripts/image_pipeline; LOG=/home/paperspace/logs/tenrows_colmap_${BLK}_sfm.log
rm -rf $W; mkdir -p $W/images $W/sparse
python3 -c "
import json, shutil
from pathlib import Path
t = json.load(open('$B/$BLK/transforms.json'))
for f in t['frames']: shutil.copy(f['file_path'], '$W/images/' + Path(f['file_path']).name)
print('[$BLK] copied', len(t['frames']))"
export LD_LIBRARY_PATH=$GLD; t0=$(date +%s)
$CC feature_extractor --database_path $W/database.db --image_path $W/images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --ImageReader.camera_params "529.4046769303875,529.6521348953188,647.1984132364281,354.6428014457265,0,0,0,0" --FeatureExtraction.use_gpu 1 > $LOG 2>&1 || { echo "[$BLK] FEAT FAILED"; exit 1; }
$CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $LOG 2>&1 || { echo "[$BLK] MATCH FAILED"; exit 1; }
$CC mapper --database_path $W/database.db --image_path $W/images --output_path $W/sparse --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 >> $LOG 2>&1 || { echo "[$BLK] MAPPER FAILED"; exit 1; }
unset LD_LIBRARY_PATH; echo "[$BLK] SfM $(( $(date +%s)-t0 ))s"
python3 $IPL/colmap_to_nerfstudio.py $W > /dev/null 2>&1
python3 - << PY
import json, numpy as np
from pathlib import Path
W = Path("$W"); B = Path("$B"); FLIP = np.diag([1., -1., -1., 1.])
cm = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) @ FLIP for f in json.loads((W/"transforms.json").read_text())["frames"]}
ours = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in json.loads((B/"$BLK/transforms.json").read_text())["frames"]}
names = [n for n in ours if n in cm]; P = np.array([cm[n][:3, 3] for n in names]); Q = np.array([ours[n][:3, 3] for n in names])
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq; U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rr = (U @ D @ Vt).T; s = (S*np.diag(D)).sum()/(X**2).sum()
res = np.linalg.norm(Q - (s*(Rr@P.T).T + (mq - s*Rr@mp)), axis=1)
R = np.array([ours[n][:3, :3] for n in names]); t = np.array([ours[n][:3, 3] for n in names]); trav = np.einsum("nji,nj->ni", R[:-1], t[1:]-t[:-1]); trav = (trav/np.linalg.norm(trav,axis=1,keepdims=True)).mean(0)
print(f"[$BLK] registered {len(cm)}/{len(ours)}; Sim(3) scale {s:.3f}, position p50 {np.percentile(res,50):.3f} m; travel dir in our cam axes {np.round(trav,2)}; world travel dir {np.round((t[-1]-t[0])/np.linalg.norm(t[-1]-t[0]),2)}")
for k, lab in enumerate(("x", "y", "z")):
    d = [float(ours[n][:3, k] @ (Rr @ cm[n][:3, k])) for n in names]; print(f"[$BLK] camera {lab}: ours·COLMAP = {np.mean(d):+.3f} (p10 {np.percentile(d,10):+.3f}, p90 {np.percentile(d,90):+.3f})")
PY
