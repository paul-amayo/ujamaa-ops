echo "=== STAGE D: lidar init + depth ==="
[ -f "$BD/init_lidar.ply" ] || [ -f "$BD/init_da3.ply" ] || \
pixi run --manifest-path $IS_PIXI python src/scripts/lidar_init_per_block.py \
  --block-dir "$BD" --root "$ROOT" --pad-x 7.5 --cross-row-median || exit 1
[ -f "$BD/lidar_depth_morph.npz" ] || \
pixi run --manifest-path $NS_PIXI python src/scripts/build_block_lidar_depth.py \
  --block-dir "$BD" --root "$ROOT" || exit 1
echo "=== STAGE E: pose swap + train ==="
[ -f "$BD/transforms.json.orig" ] || cp "$BD/transforms.json" "$BD/transforms.json.orig"
$PYNS - "$BD" "$ROOT" <<'PY' || exit 1
import json, numpy as np, sys
from pathlib import Path
B=Path(sys.argv[1]); ROOT=Path(sys.argv[2])
d=json.loads((B/"transforms.json.orig").read_text())
lio={int(p["image_idx"]):p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.])
miss=0
for f in d["frames"]:
    K=int(Path(f["file_path"]).stem.split('_')[1])
    if K in lio: f["transform_matrix"]=(np.asarray(lio[K])@CV2GL).tolist()
    else: miss+=1
(B/"transforms.json").write_text(json.dumps(d,indent=2))
print(f'pose swap done, missing {miss}')
PY
OUT=$BD/splat_runs_P1; mkdir -p "$OUT"
echo "n" | \
  HIGH_EMBEDDER_CKPT=$EMB \
  LIDAR_DEPTH_NPZ="$BD/lidar_depth_morph.npz" \
  LIDAR_DEPTH_LAMBDA=0.5 MAX_JOBS=4 \
  pixi run --manifest-path $NS_PIXI ns-train high \
    --max-num-iterations 20000 --vis tensorboard \
    --output-dir "$OUT" --experiment-name pilot_004 \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.background-color random \
    --pipeline.model.sky-loss-lambda 0.0 \
    --pipeline.model.output-depth-during-training True \
    nerfstudio-data --data "$BD" || exit 1
CFG=$(find "$OUT" -name config.yml | head -1)
pixi run --manifest-path $NS_PIXI ns-export gaussian-splat \
  --load-config "$CFG" --output-dir "$OUT/exported" --save-world-frame True \
  && touch "$OUT/DONE"
echo "=== PILOT-DONE ==="
