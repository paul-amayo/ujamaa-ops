#!/bin/bash
# WEEKEND QUEUE (queued Fri 2026-07-31, runs behind klap_psnr_sweep):
#   R  compiler repro     S4 recipe but supervision/strict_fruit_v1 id maps —
#                         gates must match S4 => id-path becomes canon
#   N  DINOv2 pre-flight  Dec-2025 -> April retrieval (Sankofa localisation)
#   P  strict params      S4 recipe + anchor 15 | + tau 5 (one var each)
#   F  klapmuts fleet     9 remaining blocks, sweep-winning config, masks+
#                         stages C/D per block, 20k, export ply
# Stages isolated: a failure logs FAIL and later stages still run.
set -x
ARU=/home/paperspace/code/aru_sil_core
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
IS_PIXI=$ARU/src/thirdparty/InstantSplat/pixi.toml
PYNS=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python
SCR=$ARU/src/scripts
LOG=/home/paperspace/logs
D04=/home/paperspace/data/citrus_all/04_13D_Jackal
CB=$D04/blocks_ns/lio_row6F/block_001
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
KROOT=/home/paperspace/data/klapmuts
KEMB=/home/paperspace/data/high/nerf/klapmuts_v1/ckpts/model_best.pth
DEADLINE=$(date -d "next saturday 22:00" +%s 2>/dev/null || date -d "tomorrow 22:00" +%s)

echo "=== WEEKEND QUEUE: waiting for sweep sentinel (deadline Sat 22:00) ==="
until grep -q "SWEEP-ALL-DONE" $LOG/klap_psnr_sweep.log 2>/dev/null \
      || [ "$(date +%s)" -gt "$DEADLINE" ]; do sleep 300; done
grep -q "SWEEP-ALL-DONE" $LOG/klap_psnr_sweep.log 2>/dev/null \
  || echo "WARN: deadline hit before sweep sentinel — proceeding when GPU free"
while ps -eo cmd | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 60; done
echo "=== WEEKEND START $(date) ==="

train_citrus () {  # name sem_dir extra_flags...
  local NAME=$1 SEM=$2; shift 2
  echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
    LIDAR_DEPTH_NPZ=$CB/lidar_depth_morph.npz \
    pixi run --manifest-path $NS_PIXI ns-train high \
      --data $CB --output-dir $CB/splat_runs_WK --experiment-name $NAME \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.output-depth-during-training True \
      --pipeline.model.high-loss-weight 1.0 \
      --pipeline.model.high-loss-fruit-weight 10.0 \
      --pipeline.model.fruit-protect True \
      --pipeline.datamanager.semantic-dir $SEM \
      "$@" \
      --max-num-iterations 20001 --steps-per-save 9998 \
      --vis tensorboard nerfstudio-data || { echo "RUN-FAIL $NAME"; return 1; }
  echo "RUN-DONE $NAME"
  local CFG=$(ls -t $CB/splat_runs_WK/$NAME/high/*/config.yml | head -1)
  sed "s|CFG = Path('.*')|CFG = Path('$CFG')|" $LOG/s2_gates_template.py > $LOG/wk_gates_$NAME.py
  cd $ARU && HIGH_EMBEDDER_CKPT=$EMB \
    pixi run --manifest-path $NS_PIXI python $LOG/wk_gates_$NAME.py 2>&1 \
    | grep -aE "RENDERED|fruit pixels|tree  pixels|CROSS-LEVEL|choosing" \
    | sed "s/^/GATES $NAME /"
  cd /home/paperspace/code
}

echo "=== STAGE R: compiler repro (id-path) ==="
train_citrus s4r_idpath $CB/supervision/strict_fruit_v1 || true

echo "=== STAGE N: DINOv2 Dec->April pre-flight ==="
pixi run --manifest-path $NS_PIXI python $LOG/dinov2_preflight.py \
  2>&1 | grep -aE "\[dinov2\]|\[stats\]|\[done\]|Error|Traceback" || echo "RUN-FAIL dinov2"

echo "=== STAGE P: strict-label parameter probe ==="
train_citrus s5_anchor15 $CB/semantic_v2_S --pipeline.model.fruit-anchor-weight 15.0 || true
train_citrus s5_tau5     $CB/semantic_v2_S --pipeline.model.fruit-protect-tau 5.0 || true

echo "=== STAGE F: klapmuts fleet (sweep-winning config) ==="
WINNER=$($PYNS - <<'PY'
import re
best, cfg = -1, "base"
try:
    for ln in open("/home/paperspace/logs/klap_psnr_sweep.log", errors="replace"):
        m = re.match(r"PSNR (sw_\w+) (\S+) = ([\d.]+)", ln.strip())
        if not m or m.group(1) not in ("sw_base", "sw_nosky", "sw_nodepth"):
            continue
        tag = m.group(2).lower()
        if "psnr" in tag and ("fg" in tag or "eval" in tag):
            v = float(m.group(3))
            if v > best:
                best, cfg = v, m.group(1)[3:]
except FileNotFoundError:
    pass
print(cfg)
PY
)
echo "FLEET winner config: $WINNER"
SKYL=1.0; DEPTH=1
[ "$WINNER" = "nosky" ] && SKYL=0.0
[ "$WINNER" = "nodepth" ] && DEPTH=0

for N in 000 001 002 003 005 006 007 008 009; do
  BD=$KROOT/blocks_ns/lio_row/block_$N
  echo "=== FLEET block_$N ==="
  ( set -e
    $PYNS - "$BD" <<'PY'
import json, sys
from pathlib import Path
B = Path(sys.argv[1])
names = [Path(f["file_path"]).name
         for f in json.loads((B/"transforms.json").read_text())["frames"]]
Path("/tmp/fleet_frames.json").write_text(json.dumps(sorted(names)))
PY
    cd $ARU
    pixi run --manifest-path $NS_PIXI python $SCR/build_sky_masks.py \
      --data-dir $KROOT --keep-frames /tmp/fleet_frames.json
    python3 $SCR/build_fg_masks.py --data-dir $KROOT --keep-frames /tmp/fleet_frames.json
    [ -f "$BD/semantic_v2_B/palette.json" ] || \
      python3 $SCR/save_semantic_pngs_fast.py --block-dir $BD --root $KROOT \
        --semantic-monolithic $KROOT/filtered_semantic_v2.monolithic
    [ -f "$BD/init_da3.ply" ] || \
      pixi run --manifest-path $IS_PIXI python $SCR/lidar_init_per_block.py \
        --block-dir $BD --root $KROOT --pad-x 7.5 --cross-row-median
    if [ "$DEPTH" = 1 ]; then
      [ -f "$BD/lidar_depth_morph.npz" ] || \
        pixi run --manifest-path $NS_PIXI python $SCR/build_block_lidar_depth.py \
          --block-dir $BD --root $KROOT
    fi
    $PYNS - "$BD" "$KROOT" <<'PY'
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
    ENVV=(HIGH_EMBEDDER_CKPT=$KEMB MAX_JOBS=4)
    [ "$DEPTH" = 1 ] && ENVV+=(LIDAR_DEPTH_NPZ=$BD/lidar_depth_morph.npz LIDAR_DEPTH_LAMBDA=0.5)
    echo "n" | env "${ENVV[@]}" \
      pixi run --manifest-path $NS_PIXI ns-train high \
        --max-num-iterations 20000 --vis tensorboard \
        --output-dir $BD/splat_runs_F1 --experiment-name fleet_$N \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.use-scale-regularization True \
        --pipeline.model.background-color random \
        --pipeline.model.sky-loss-lambda $SKYL \
        --pipeline.model.output-depth-during-training True \
        nerfstudio-data --data $BD --eval-mode interval --eval-interval 10
    CFG=$(find $BD/splat_runs_F1 -name config.yml | head -1)
    pixi run --manifest-path $NS_PIXI ns-export gaussian-splat \
      --load-config "$CFG" --output-dir $BD/splat_runs_F1/exported \
      --save-world-frame True
    echo "FLEET-DONE block_$N"
  ) || echo "FLEET-FAIL block_$N"
done
echo "=== WEEKEND-ALL-DONE $(date) ==="
