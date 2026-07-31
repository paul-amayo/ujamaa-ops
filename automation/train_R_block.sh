#!/usr/bin/env bash
# R-retrain one 01_13B lio_row block: targets rebuilt via the fixed config-aware
# embedder loader (high@d091981), matched embedder 01_13B_v1. Recipe recovered
# from the crashed 2026-07-13_132409 R run's config.yml (= C-config + depth-sup).
# Usage: train_R_block.sh <NNN>
set -e
N=$1
ROOT=/home/paperspace/data/citrus_all/01_13B_Jackal
B=$ROOT/blocks_ns/lio_row/block_$N
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
ARU=/home/paperspace/code/aru_sil_core
OUT=$B/splat_runs_R_20k
Q=/home/paperspace/code/automation/quarantine.sh

# stale artifacts out of the way (never delete)
[ -d "$OUT" ] && $Q "$OUT"
for c in "$ARU"/outputs/block_${N}* "$ARU"/outputs/data_block_${N}*; do
  [ -e "$c" ] && $Q "$c"; done
mkdir -p "$OUT"

# swap transforms to LIO + CV->GL (verified raw-CV state 2026-07-13)
cp "$B/transforms.json" "$B/transforms.json.refined"
python3 - <<PY
import json, numpy as np
from pathlib import Path
B=Path("$B"); ROOT=Path("$ROOT")
d=json.loads((B/"transforms.json").read_text())
lio={p["image_name"]:p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.]); n=0
for f in d["frames"]:
    name=Path(f["file_path"]).name
    if name in lio:
        f["transform_matrix"]=(np.asarray(lio[name])@CV2GL).tolist(); n+=1
d["ply_file_path"]="init_da3.ply"
(B/"transforms.json").write_text(json.dumps(d,indent=2))
print(f"swapped {n}/{len(d['frames'])}")
PY

restore() { mv -f "$B/transforms.json.refined" "$B/transforms.json"; }
trap restore EXIT

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
    --experiment-name "R20k_block_${N}" \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.background-color black \
    --pipeline.model.sky-loss-lambda 0.05 \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.report-masked-metrics True \
    nerfstudio-data --data "$B"

CFG=$(find "$OUT" -name config.yml | head -1)
pixi run --manifest-path "$NS_PIXI" ns-export gaussian-splat \
  --load-config "$CFG" --output-dir "$OUT/exported" --save-world-frame True
echo "R-BLOCK $N DONE"
