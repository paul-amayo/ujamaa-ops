#!/bin/bash
# IMG_7999_s0 flag test: same recipe train command (RGB mode) but the resolution schedule finishes inside the run
# (schedule 1500 -> /4 until 1500, /2 until 3000, full res after), ITERS iterations. Output beside the recipe's run.
set -uo pipefail
SD=/home/paperspace/data/image_farm/IMG_7999_s0
BD=$SD/blocks_ns/lio_arc_size15.0_ov0.10_kf20cm_dedup/block_000
NS=/home/paperspace/code/nerf_new
ITERS=${1:-5000}; SCHED=${2:-1500}; TAG=${3:-sched${SCHED}_it${ITERS}}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] sched-test $*"; }
rm -rf "$BD/splat_runs_$TAG" "$NS/outputs/block_000" /home/paperspace/code/outputs/block_000
say "train $TAG"
( cd "$NS" && echo n | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run ns-train high \
  --max-num-iterations "$ITERS" --vis tensorboard \
  --output-dir "$BD/splat_runs_$TAG" --experiment-name "IMG_7999_s0_$TAG" \
  --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
  --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 \
  --pipeline.model.resolution-schedule "$SCHED" --pipeline.model.stop-split-at "${STOP_SPLIT:-15000}" \
  --pipeline.model.cull-alpha-thresh 0.01 \
  --pipeline.model.cull-scale-thresh 0.3 \
  --pipeline.model.densify-grad-thresh 0.0006 \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.background-color black \
  --pipeline.model.report-masked-metrics True \
  nerfstudio-data --data "$BD" ) > /home/paperspace/logs/img7999_s0_$TAG.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python - "$BD/splat_runs_$TAG" << 'PY'
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
run = sorted(Path(sys.argv[1]).glob("*/*/*"))[-1]
ea = EventAccumulator(str(run), size_guidance={"scalars": 0, "images": 0}); ea.Reload()
for tag in ["Train Metrics Dict/psnr", "Eval Images Metrics/psnr"]:
    if tag in ea.Tags().get("scalars", []):
        s = ea.Scalars(tag); v = [x.value for x in s]; st = [x.step for x in s]
        lastk = [x for a, x in zip(st, v) if a >= st[-1] - 1000]
        print(f"PSNR {tag}: last={v[-1]:.2f} mean-last-1k={np.mean(lastk):.2f}")
g = ea.Scalars("Train Metrics Dict/gaussian_count"); print(f"gaussians final {int(g[-1].value)}")
for tag in ea.Tags()["images"]:
    last = ea.Images(tag)[-1]; out = Path(f"/home/paperspace/logs/img7999s0_{Path(sys.argv[1]).name}_step{last.step}.png"); out.write_bytes(last.encoded_image_string); print("saved", out)
PY
) 2>/dev/null | grep -E "PSNR|gaussians|saved"
say "DONE $TAG"
