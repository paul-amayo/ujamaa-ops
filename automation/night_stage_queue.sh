#!/bin/bash
# NIGHT QUEUE (Paul, 2026-08-05): first the FIXED two-stage rerun (v3), then
# STAGE 1 (scene-baseline appearance, treelod_bg00_v1 recipe) for every row
# block, surveys in Paul's order: 04, 05, 01, 02, 03.
#
# Inventory at queue time: 04 lio_row6F 6 | 05 lio_row 15 | 01 lio_row 35 |
# 02 none (redownload pending, skipped loudly) | 03 lio_row_halves 80.
# ~75 min/block: one night covers v3 + ~7 blocks; the queue is RESUMABLE —
# blocks with an existing stage1_bg00 run are skipped, so relaunching this
# script continues where the night ended.
#
# Recipe per block = scene.json baseline (bg00): tree-weighted L1 (bg 0.0)
# where the block has compiled supervision (TREE_WEIGHT_DIR set only then;
# loader disables itself loudly otherwise), sky loss 1.0 (no-op where no
# masks), stop-split 6000, 15k steps, NO depth, canary armed.
set -x
# QUEUE HOLD: touch /home/paperspace/logs/QUEUE_HOLD to make any invocation
# (incl. the auto-chain from stage2_fruitchild.sh) exit without training.
if [ -f /home/paperspace/logs/QUEUE_HOLD ]; then
  echo "NIGHT-HOLD: queue on hold ($(cat /home/paperspace/logs/QUEUE_HOLD))"
  exit 0
fi
# KLAPMUTS-FIRST gate (Paul 2026-08-07): if the flag exists, the klapmuts
# queue runs first and chains back into this script when done.
if [ -f /home/paperspace/logs/KLAP_FIRST ]; then
  exec /home/paperspace/code/automation/klap_first_queue.sh
fi
cd /home/paperspace/code/nerf_new
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
ROOT=/home/paperspace/data/citrus_all

# ---- 0. two-stage v3 with the freeze-after-load fix ----
BD=$ROOT/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
SUP04=$ROOT/04_13D_Jackal/blocks_ns/lio_row6F/block_001/supervision/strict_tree_v2
if [ ! -d "$BD/splat_runs_FEATFIX/stage2_frozen_v3" ]; then
  T0=$(date +%s)
  echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB CANARY_EVERY=1000 \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_frozen_v3 \
      --load-dir $BD/stage2_init/nerfstudio_models \
      --pipeline.model.freeze-geometry True \
      --pipeline.model.high-loss-weight 1.0 \
      --pipeline.datamanager.semantic-dir $SUP04 \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.sky-loss-lambda 1.0 \
      --pipeline.model.report-masked-metrics True \
      --max-num-iterations 20001 --steps-per-save 19998 \
      --vis tensorboard nerfstudio-data \
      --eval-mode interval --eval-interval 10 \
      && echo "NIGHT-DONE stage2_frozen_v3 in $(( ($(date +%s)-T0)/60 )) min" \
      || echo "NIGHT-FAIL stage2_frozen_v3"
fi

# ---- 1. stage-1 appearance for every row block, survey order 04,05,01,02,03 ----
for SPEC in "04_13D_Jackal:lio_row6F" "05_13D_Jackal:lio_row" \
            "01_13B_Jackal:lio_row" "02_13B_Jackal:lio_row" \
            "03_13B_Jackal:lio_row_halves"; do
  S=${SPEC%%:*}; SCHEME=${SPEC##*:}
  if ! ls -d $ROOT/$S/blocks_ns/$SCHEME/block_[0-9][0-9][0-9] >/dev/null 2>&1; then
    echo "NIGHT-SKIP $S: no $SCHEME blocks (02 = redownload pending)"
    continue
  fi
  for B in $ROOT/$S/blocks_ns/$SCHEME/block_[0-9][0-9][0-9]; do
    N=$(basename $B)
    [ -f "$B/transforms.json" ] || { echo "NIGHT-SKIP $S/$N: no transforms"; continue; }
    # done = a saved CHECKPOINT exists, not just the run dir — a failed
    # launch creates the dir in seconds (2026-08-06 crash-loop: an orphaned
    # trainer held the GPU and 6 blocks "completed" in 30 s each)
    if ls $B/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
      echo "NIGHT-SKIP $S/$N: done"; continue
    fi
    TW=""
    [ -f "$B/supervision/strict_tree_v2/manifest.json" ] && TW=$B/supervision/strict_tree_v2
    # mask consistency fixup (2026-08-08): nerfstudio asserts masks are
    # all-or-none per split; 01/03-era transforms carry mask_path for only
    # SOME frames -> instant AssertionError crash-loop. If coverage is
    # partial or any file is missing, strip mask_path everywhere (.orig kept).
    python3 - "$B" << 'PYFIX'
import json, sys
from pathlib import Path
B = Path(sys.argv[1]); tj = B / "transforms.json"
d = json.loads(tj.read_text())
fr = d.get("frames", [])
withm = [f for f in fr if f.get("mask_path")]
ok = withm and len(withm) == len(fr) and all(
    (B / f["mask_path"]).exists() or Path(f["mask_path"]).exists() for f in withm)
if withm and not ok:
    bak = B / "transforms.json.premaskfix"
    if not bak.exists():
        bak.write_text(tj.read_text())
    for f in fr:
        f.pop("mask_path", None)
    tj.write_text(json.dumps(d, indent=1))
    print(f"NIGHT-MASKFIX {B.name}: stripped partial mask_path "
          f"({len(withm)}/{len(fr)} frames had one)")
PYFIX
    T0=$(date +%s)
    echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
      TREE_WEIGHT_DIR=$TW TREE_WEIGHT_BG=0.0 \
      pixi run ns-train high \
        --data $B --output-dir $B/splat_runs_STAGE1 --experiment-name stage1_bg00 \
        --pipeline.model.enable-high-features False \
        --pipeline.model.high-loss-weight 0.0 \
        --pipeline.datamanager.semantic-dir $EMPTY \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.stop-split-at 6000 \
        --pipeline.model.sky-loss-lambda 1.0 \
        --pipeline.model.report-masked-metrics True \
        --max-num-iterations 15001 --steps-per-save 14998 \
        --vis tensorboard nerfstudio-data \
        --eval-mode interval --eval-interval 10 \
        && echo "NIGHT-DONE $S/$N in $(( ($(date +%s)-T0)/60 )) min" \
        || echo "NIGHT-FAIL $S/$N"
  done
done
echo "=== NIGHT-QUEUE-ALL-DONE $(date) ==="
