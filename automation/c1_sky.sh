#!/usr/bin/env bash
# C1 sky-objective ablation on 01/lio_row block_000 (baseline arm = existing
# R_20k run: black bg, sky-loss 0.05). Variants target the unsupervised-sky
# dark-floater problem. Usage: c1_sky.sh
set -e
ROOT=/home/paperspace/data/citrus_all/01_13B_Jackal
B=$ROOT/blocks_ns/lio_row/block_000
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
ARU=/home/paperspace/code/aru_sil_core
Q=/home/paperspace/code/automation/quarantine.sh

cp "$B/transforms.json" "$B/transforms.json.c1bak"
python3 - <<PY
import json, numpy as np
from pathlib import Path
B=Path("$B"); ROOT=Path("$ROOT")
d=json.loads((B/"transforms.json").read_text())
lio={p["image_name"]:p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.])
for f in d["frames"]:
    n=Path(f["file_path"]).name
    if n in lio: f["transform_matrix"]=(np.asarray(lio[n])@CV2GL).tolist()
d["ply_file_path"]="init_da3.ply"
(B/"transforms.json").write_text(json.dumps(d,indent=2))
PY
restore() { mv -f "$B/transforms.json.c1bak" "$B/transforms.json"; }
trap restore EXIT

run_variant() { # name bg skylam
  NAME=$1; BG=$2; LAM=$3
  OUT=$B/splat_runs_c1_$NAME
  [ -d "$OUT" ] && $Q "$OUT"
  mkdir -p "$OUT"
  cd "$ARU"
  echo "n" | \
    HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/01_13B_v1/ckpts/model_best.pth \
    LIDAR_DEPTH_NPZ="$B/lidar_depth_morph.npz" \
    LIDAR_DEPTH_LAMBDA=0.5 \
    MAX_JOBS=4 \
    pixi run --manifest-path "$NS_PIXI" ns-train high \
      --max-num-iterations 20000 \
      --vis tensorboard \
      --output-dir "$OUT" \
      --experiment-name "c1_${NAME}_block_000" \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.cull-alpha-thresh 0.005 \
      --pipeline.model.use-scale-regularization True \
      --pipeline.model.background-color $BG \
      --pipeline.model.sky-loss-lambda $LAM \
      --pipeline.model.output-depth-during-training True \
      --pipeline.model.report-masked-metrics True \
      nerfstudio-data --data "$B"
  echo "C1 VARIANT $NAME DONE"
}

run_variant randbg005 random 0.05
run_variant black020  black  0.20
run_variant randbg020 random 0.20
echo "C1 ALL DONE"
