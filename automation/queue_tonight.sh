#!/bin/bash
# Master queue 2026-07-31: pilot(D-E) -> S3a train -> S3a gates -> klapmuts fleet.
# Durable paths only. Each stage logs to ~/logs and hard-stops on failure.
set -x
LOGS=/home/paperspace/logs
B004=/home/paperspace/data/klapmuts/blocks_ns/lio_row/block_004

# 1. wait for the pilot (already running via klap_pilot_de_run.sh)
until grep -qaE "PILOT-DONE" $LOGS/klap_pilot_de2.log 2>/dev/null; do
  grep -qaE "Traceback|FATAL" $LOGS/klap_pilot_de2.log 2>/dev/null && \
    { echo "QUEUE: pilot failed — stopping"; exit 1; }
  sleep 300
done
echo "QUEUE: pilot done"

# 2. S3a — protected fruit with densification-surviving tallies (citrus b001)
CB=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001
cd /home/paperspace/code/nerf_new
echo "n" | \
  MAX_JOBS=4 \
  HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth \
  LIDAR_DEPTH_NPZ=$CB/lidar_depth_morph.npz \
  pixi run ns-train high \
    --data $CB --output-dir $CB/splat_runs_S3 --experiment-name fruit_s3a \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight 10.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.datamanager.semantic-dir $CB/semantic_v2_C \
    --max-num-iterations 20001 --steps-per-save 9998 \
    --vis tensorboard nerfstudio-data || { echo "QUEUE: S3a train failed"; exit 1; }
echo "QUEUE: S3a trained"

# 3. S3a gates
S3CFG=$(ls -t $CB/splat_runs_S3/fruit_s3a/high/*/config.yml | head -1)
sed "s|CFG = Path('.*')|CFG = Path('$S3CFG')|" /tmp/s1_gates.py > /home/paperspace/logs/s3_gates.py 2>/dev/null || \
  cp /tmp/s2_gates.py /home/paperspace/logs/s3_gates.py
cd /home/paperspace/code/aru_sil_core
HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth \
pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml \
  python /home/paperspace/logs/s3_gates.py > $LOGS/s3_gates.log 2>&1
grep -aE "RENDERED|CROSS-LEVEL|choosing" $LOGS/s3_gates.log
echo "QUEUE: S3a gates done"

# 4. klapmuts fleet — remaining 9 row blocks with the FAST stage C
ROOT=/home/paperspace/data/klapmuts
EMB=/home/paperspace/data/high/nerf/klapmuts_v1/ckpts/model_best.pth
ARU=/home/paperspace/code/aru_sil_core
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
IS_PIXI=$ARU/src/thirdparty/InstantSplat/pixi.toml
PYNS=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python
for N in 000 001 002 003 005 006 007 008 009; do
  BD=$ROOT/blocks_ns/lio_row/block_$N
  echo "QUEUE: fleet block_$N"
  cd $ARU
  [ -f "$BD/semantic_v2_B/palette.json" ] || \
  $PYNS src/scripts/save_semantic_pngs_fast.py --block-dir "$BD" --root "$ROOT" \
    --semantic-monolithic "$ROOT/filtered_semantic_v2.monolithic" \
    --global-ids "$ROOT/sam3_sack/global_ids.json" || continue
  [ -f "$BD/init_da3.ply" ] || \
  pixi run --manifest-path $IS_PIXI python src/scripts/lidar_init_per_block.py \
    --block-dir "$BD" --root "$ROOT" --pad-x 7.5 --cross-row-median || continue
  [ -f "$BD/lidar_depth_morph.npz" ] || \
  pixi run --manifest-path $NS_PIXI python src/scripts/build_block_lidar_depth.py \
    --block-dir "$BD" --root "$ROOT" || continue
  [ -f "$BD/transforms.json.orig" ] || cp "$BD/transforms.json" "$BD/transforms.json.orig"
  $PYNS - "$BD" "$ROOT" <<'PY'
import json, numpy as np, sys
from pathlib import Path
B=Path(sys.argv[1]); ROOT=Path(sys.argv[2])
d=json.loads((B/"transforms.json.orig").read_text())
lio={int(p["image_idx"]):p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.])
for f in d["frames"]:
    K=int(Path(f["file_path"]).stem.split('_')[1])
    if K in lio: f["transform_matrix"]=(np.asarray(lio[K])@CV2GL).tolist()
(B/"transforms.json").write_text(json.dumps(d,indent=2))
PY
  OUT=$BD/splat_runs_P1; mkdir -p "$OUT"
  echo "n" | HIGH_EMBEDDER_CKPT=$EMB LIDAR_DEPTH_NPZ="$BD/lidar_depth_morph.npz" \
    LIDAR_DEPTH_LAMBDA=0.5 MAX_JOBS=4 \
    pixi run --manifest-path $NS_PIXI ns-train high \
      --max-num-iterations 20000 --vis tensorboard \
      --output-dir "$OUT" --experiment-name "row_$N" \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.cull-alpha-thresh 0.005 \
      --pipeline.model.use-scale-regularization True \
      --pipeline.model.background-color random \
      --pipeline.model.sky-loss-lambda 0.0 \
      --pipeline.model.output-depth-during-training True \
      nerfstudio-data --data "$BD" || { echo "QUEUE: block_$N train failed"; continue; }
  CFG=$(find "$OUT" -name config.yml | head -1)
  pixi run --manifest-path $NS_PIXI ns-export gaussian-splat \
    --load-config "$CFG" --output-dir "$OUT/exported" --save-world-frame True \
    && touch "$OUT/DONE"
done
echo "QUEUE-ALL-DONE"
