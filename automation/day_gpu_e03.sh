#!/usr/bin/env bash
# E-03 DAY BATCH: build 03_13B half-row blocks, then per block: semantic PNGs ->
# lidar init -> depth npz -> pose swap -> train (G2-winner config: random bg,
# sky-loss 0.20; paired embedder 03_13B_v1; depth-sup) -> export -> restore.
# One approval launches multi-day work. Skips per-block steps already done, so
# it is safe to re-launch after any interruption.
set -e
ROOT=/home/paperspace/data/citrus_all/03_13B_Jackal
CFGDIR=$ROOT/blocks_ns/lio_row_halves
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
IS_PIXI=/home/paperspace/code/aru_sil_core/src/thirdparty/InstantSplat/pixi.toml
ARU=/home/paperspace/code/aru_sil_core
EMB=/home/paperspace/data/high/nerf/03_13B_v2G/ckpts/model_best.pth  # graph-template embedder, validated 99.26% retrieval 2026-07-22
Q=/home/paperspace/code/automation/quarantine.sh

# disk guard
free_gb() { df --output=avail -BG / | tail -1 | tr -dc 0-9; }

[ -d "$CFGDIR" ] || pixi run --manifest-path "$NS_PIXI" \
  python /home/paperspace/code/automation/build_row_halves.py "$ROOT" lio_row_halves

for BD in "$CFGDIR"/block_*; do
  N=$(basename "$BD")
  if [ "$(free_gb)" -lt 120 ]; then echo "DISK GUARD TRIPPED ($(free_gb)G) — stopping"; break; fi
  if [ -f "$BD/splat_runs_E/DONE" ]; then echo "[$N] already done, skip"; continue; fi
  echo "=== $N: begin $(date -u +%H:%M) ==="

  [ -f "$BD/semantic_v2_B/palette.json" ] || \
    pixi run --manifest-path "$NS_PIXI" python "$ARU/src/scripts/save_filtered_semantic_pngs.py" \
      --block-dir "$BD" \
      --semantic-monolithic "$ROOT/filtered_semantic_v2_B.monolithic" \
      --marker-monolithic "$ROOT/scene_graph/markers_v2_B.monolithic"

  [ -f "$BD/init_da3.ply" ] || \
    pixi run --manifest-path "$IS_PIXI" python "$ARU/src/scripts/lidar_init_per_block.py" \
      --block-dir "$BD" --root "$ROOT" --pad-x 7.5 --cross-row-median

  [ -f "$BD/lidar_depth_morph.npz" ] || \
    pixi run --manifest-path "$NS_PIXI" python "$ARU/src/scripts/build_block_lidar_depth.py" \
      --block-dir "$BD" --root "$ROOT"

  # PREP_ONLY=1: stop after sem-PNGs/init/depth (embedder-independent stages);
  # training resumes later once the graph-trained embedder is validated.
  if [ "${PREP_ONLY:-0}" = "1" ]; then echo "[$N] prep complete (PREP_ONLY)"; continue; fi

  # swap poses to GL for training (blocks are written raw-CV by the generator)
  cp "$BD/transforms.json" "$BD/transforms.json.orig"
  python3 - "$BD" "$ROOT" <<'PY'
import json, numpy as np, sys
from pathlib import Path
B=Path(sys.argv[1]); ROOT=Path(sys.argv[2])
d=json.loads((B/"transforms.json").read_text())
lio={p["image_name"]:p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.])
for f in d["frames"]:
    n=Path(f["file_path"]).name
    if n in lio: f["transform_matrix"]=(np.asarray(lio[n])@CV2GL).tolist()
(B/"transforms.json").write_text(json.dumps(d,indent=2))
PY

  OUT=$BD/splat_runs_E
  [ -d "$OUT" ] && $Q "$OUT"; mkdir -p "$OUT"
  cd "$ARU"
  set +e
  echo "n" | \
    HIGH_EMBEDDER_CKPT=$EMB \
    LIDAR_DEPTH_NPZ="$BD/lidar_depth_morph.npz" \
    LIDAR_DEPTH_LAMBDA=0.5 \
    MAX_JOBS=4 \
    pixi run --manifest-path "$NS_PIXI" ns-train high \
      --max-num-iterations 20000 \
      --vis tensorboard \
      --output-dir "$OUT" \
      --experiment-name "E_${N}" \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.cull-alpha-thresh 0.005 \
      --pipeline.model.use-scale-regularization True \
      --pipeline.model.background-color random \
      --pipeline.model.sky-loss-lambda 0.20 \
      --pipeline.model.output-depth-during-training True \
      --pipeline.model.report-masked-metrics True \
      nerfstudio-data --data "$BD"
  TRAIN_RC=$?
  set -e
  if [ $TRAIN_RC -eq 0 ]; then
    CFG=$(find "$OUT" -name config.yml | head -1)
    pixi run --manifest-path "$NS_PIXI" ns-export gaussian-splat \
      --load-config "$CFG" --output-dir "$OUT/exported" --save-world-frame True \
      && touch "$OUT/DONE"
  else
    echo "[$N] TRAIN FAILED rc=$TRAIN_RC — continuing to next block"
  fi
  mv -f "$BD/transforms.json.orig" "$BD/transforms.json"
  echo "=== $N: end $(date -u +%H:%M) rc=$TRAIN_RC ==="
done
echo "E-03 DAY BATCH COMPLETE $(date -u)"
