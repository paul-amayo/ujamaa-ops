#!/bin/bash
# After SfM-all: re-map blocks whose incremental COLMAP model is bad (coverage <90% or axis dot <0.95 or p50 >0.3 m)
# with the GPU GLOMAP global mapper on the same database, then re-run the axis check.
OUT=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/tasks/bpxrgxwqf.output
until grep -q "SFM-ALL DONE" $OUT 2>/dev/null; do sleep 60; done
T=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili; B=$T/blocks_ns/lio_row100
GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap; IPL=/home/paperspace/code/aru_sil_core/src/scripts/image_pipeline
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
BAD=$(python3 - << 'PY'
import re
txt = open("/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/tasks/bpxrgxwqf.output").read()
bad = []
for m in re.finditer(r"\[(block_\d+)\] registered (\d+)/(\d+);.*?position p50 ([\d.]+) m", txt):
    blk, a, b, p50 = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    dx = re.search(rf"\[{blk}\] camera x: ours·COLMAP = ([+-][\d.]+)", txt); dx = float(dx.group(1)) if dx else 0.0
    if a / b < 0.9 or dx < 0.95 or p50 > 0.3: bad.append(blk)
print(" ".join(bad))
PY
)
echo "[retry] blocks to re-map with GLOMAP: $BAD"
export LD_LIBRARY_PATH=$GLD
for blk in $BAD; do
  W=$T/colmap_$blk; [ -f $W/database.db ] || { echo "[retry] $blk: no database"; continue; }
  rm -rf $W/sparse $W/transforms.json $W/init.ply; mkdir -p $W/sparse; t0=$(date +%s)
  $GB mapper --database_path $W/database.db --image_path $W/images --output_path $W/sparse > /home/paperspace/logs/tenrows_glomap_$blk.log 2>&1 || { echo "[retry] $blk: GLOMAP FAILED"; continue; }
  echo "[retry] $blk: glomap $(( $(date +%s)-t0 ))s, models: $(ls $W/sparse | tr '\n' ' ')"
done
unset LD_LIBRARY_PATH
for blk in $BAD; do
  W=$T/colmap_$blk; [ -d $W/sparse/0 ] || continue
  python3 $IPL/colmap_to_nerfstudio.py $W > /dev/null 2>&1
  python3 - << PY
import json, numpy as np
from pathlib import Path
W = Path("$W"); B = Path("$B"); FLIP = np.diag([1., -1., -1., 1.])
cm = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) @ FLIP for f in json.loads((W/"transforms.json").read_text())["frames"]}
ours = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) @ FLIP for f in json.loads((B/"$blk/transforms.json").read_text())["frames"]}
names = [n for n in ours if n in cm]; P = np.array([cm[n][:3, 3] for n in names]); Q = np.array([ours[n][:3, 3] for n in names])
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq; U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rr = (U @ D @ Vt).T; s = (S*np.diag(D)).sum()/(X**2).sum()
res = np.linalg.norm(Q - (s*(Rr@P.T).T + (mq - s*Rr@mp)), axis=1); dx = np.mean([float(ours[n][:3, 0] @ (Rr @ cm[n][:3, 0])) for n in names])
print(f"[retry] $blk (glomap): registered {len(cm)}/{len(ours)}; position p50 {np.percentile(res,50):.3f} m; camera x dot {dx:+.3f}")
PY
done
echo "SFM-RETRY DONE $(date '+%H:%M:%S')"
