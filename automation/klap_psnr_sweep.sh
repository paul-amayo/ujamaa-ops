#!/bin/bash
# klapmuts PSNR-improvement sweep — queued behind S4 (waits for its sentinel).
#
# Pilot post-mortem (notebook 2026-07-29/31): eval 9.9 & declining, floater
# curtain on the camera path (unmasked sky/translucent roof), eval set was one
# frame. This sweep fixes the protocol (fg masks + eval-mode interval) and
# ablates one variable at a time around a base config:
#   base  R1: block_004 (195f), 20k steps, depth lambda 0.5, sky lambda 1.0
#   sky   R2: sky lambda 0        (P1 pilot = the no-mask-at-all datapoint)
#   depth R3: no lidar depth supervision
#   len   R5: 97-frame centre sub-block   R6: 49-frame centre sub-block
#   steps R4: 35k steps (runs LAST - longest)
# Metric: eval PSNR (+ fg-PSNR: masked metrics) from tensorboard events.
set -x
ARU=/home/paperspace/code/aru_sil_core
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
IS_PIXI=$ARU/src/thirdparty/InstantSplat/pixi.toml
PYNS=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python
ROOT=/home/paperspace/data/klapmuts
BD=$ROOT/blocks_ns/lio_row/block_004
EMB=/home/paperspace/data/high/nerf/klapmuts_v1/ckpts/model_best.pth
SCR=$ARU/src/scripts
LOG=/home/paperspace/logs

echo "=== SWEEP QUEUED: waiting for S4 sentinel ==="
until grep -q "S4-ALL-DONE" $LOG/s4_strict.log 2>/dev/null; do sleep 120; done
# belt+braces: never share the GPU with a straggler
while ps -eo cmd | grep -E "^.*ns-train" | grep -v grep > /dev/null; do sleep 60; done
echo "=== SWEEP START $(date) ==="

echo "=== STAGE M: sky -> fg masks for block_004 frames ==="
$PYNS - "$BD" <<'PY'
import json, sys
from pathlib import Path
B = Path(sys.argv[1])
names = [Path(f["file_path"]).name
         for f in json.loads((B/"transforms.json").read_text())["frames"]]
Path("/tmp/klap_sweep_frames.json").write_text(json.dumps(sorted(names)))
print(f"{len(names)} frames for masking")
PY
cd $ARU
pixi run --manifest-path $NS_PIXI python $SCR/build_sky_masks.py \
  --data-dir $ROOT --keep-frames /tmp/klap_sweep_frames.json || exit 1
python3 $SCR/build_fg_masks.py --data-dir $ROOT \
  --keep-frames /tmp/klap_sweep_frames.json || exit 1

echo "=== STAGE V: block-length variants ==="
python3 $SCR/make_block_variants.py --block-dir $BD --lengths 97 49 || exit 1

patch_masks () {  # add mask_path wherever a fg mask exists
$PYNS - "$1" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
B, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads((B/"transforms.json").read_text())
n = 0
for f in d["frames"]:
    mp = ROOT/"fg_masks"/Path(f["file_path"]).name
    if mp.exists():
        f["mask_path"] = str(mp); n += 1
(B/"transforms.json").write_text(json.dumps(d, indent=2))
print(f"[{B.name}] mask_path on {n}/{len(d['frames'])}")
PY
}
patch_masks $BD
patch_masks ${BD}_L097
patch_masks ${BD}_L049

echo "=== STAGE D: lidar init + depth for variants ==="
for V in ${BD}_L097 ${BD}_L049; do
  [ -f "$V/init_da3.ply" ] || \
  pixi run --manifest-path $IS_PIXI python $SCR/lidar_init_per_block.py \
    --block-dir "$V" --root "$ROOT" --pad-x 7.5 --cross-row-median || exit 1
  [ -f "$V/lidar_depth_morph.npz" ] || \
  pixi run --manifest-path $NS_PIXI python $SCR/build_block_lidar_depth.py \
    --block-dir "$V" --root "$ROOT" || exit 1
done

run_one () {  # name block iters depth_npz sky_lambda
  local NAME=$1 BLK=$2 ITERS=$3 DNPZ=$4 SKYL=$5
  local OUT=$BLK/splat_runs_SW
  echo "=== RUN $NAME: $(basename $BLK) iters=$ITERS depth=${DNPZ:+on} sky=$SKYL ==="
  local ENVV=(HIGH_EMBEDDER_CKPT=$EMB MAX_JOBS=4)
  [ -n "$DNPZ" ] && ENVV+=(LIDAR_DEPTH_NPZ=$DNPZ LIDAR_DEPTH_LAMBDA=0.5)
  echo "n" | env "${ENVV[@]}" \
    pixi run --manifest-path $NS_PIXI ns-train high \
      --max-num-iterations $ITERS --vis tensorboard \
      --output-dir "$OUT" --experiment-name $NAME \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.cull-alpha-thresh 0.005 \
      --pipeline.model.use-scale-regularization True \
      --pipeline.model.background-color random \
      --pipeline.model.sky-loss-lambda $SKYL \
      --pipeline.model.output-depth-during-training True \
      nerfstudio-data --data "$BLK" \
      --eval-mode interval --eval-interval 10 || { echo "RUN-FAIL $NAME"; return 1; }
  echo "=== RUN-DONE $NAME ==="
  $PYNS - "$OUT" "$NAME" <<'PY'
import sys, glob
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = sorted(glob.glob(f"{sys.argv[1]}/{sys.argv[2]}/high/*/"), reverse=True)
ea = EventAccumulator(runs[0]); ea.Reload()
for tag in ea.Tags()["scalars"]:
    lt = tag.lower()
    if "psnr" in lt:
        v = ea.Scalars(tag)[-1]
        print(f"PSNR {sys.argv[2]} {tag} = {v.value:.3f} @step {v.step}")
PY
}

run_one sw_base      $BD          20000 "$BD/lidar_depth_morph.npz"          1.0
run_one sw_nosky     $BD          20000 "$BD/lidar_depth_morph.npz"          0.0
run_one sw_nodepth   $BD          20000 ""                                   1.0
run_one sw_L097      ${BD}_L097   20000 "${BD}_L097/lidar_depth_morph.npz"   1.0
run_one sw_L049      ${BD}_L049   20000 "${BD}_L049/lidar_depth_morph.npz"   1.0
run_one sw_35k       $BD          35000 "$BD/lidar_depth_morph.npz"          1.0
echo "=== SWEEP-ALL-DONE $(date) ==="
