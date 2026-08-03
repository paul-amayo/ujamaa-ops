#!/bin/bash
# 05_13D length baseline — queued behind the citrus+klapmuts sweep.
# Prelims done 2026-08-03: lio_image_poses_kf20cm.json generated (3368 poses,
# same >=20 cm greedy rule verified against 04), 15 row blocks built
# (57-64 m each = 2.5x the 24 m cap), block_005 variants cut.
# This script does the remaining prelim (LiDAR init for the parent, ply
# copied to every variant) then runs the same splatfacto-equivalent baseline.
set -x
ARU=/home/paperspace/code/aru_sil_core
IS_PIXI=$ARU/src/thirdparty/InstantSplat/pixi.toml
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
D5=/home/paperspace/data/citrus_all/05_13D_Jackal
RB=$D5/blocks_ns/lio_row
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
until grep -qa "LENGTH-BASELINE-ALL-DONE" /home/paperspace/logs/length_baseline.log 2>/dev/null; do sleep 60; done
while ps -eo args | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 30; done
echo "=== 05 PRELIM: lidar init ==="
cd $ARU
[ -f "$RB/block_005/init_da3.ply" ] || \
  pixi run --manifest-path $IS_PIXI python $ARU/src/scripts/lidar_init_per_block.py \
    --block-dir $RB/block_005 --root $D5 --pad-x 7.5 --cross-row-median || echo "05-INIT-FAIL"
for v in L100 L075 L050; do
  [ -f "$RB/block_005/init_da3.ply" ] && cp -n $RB/block_005/init_da3.ply $RB/block_005_$v/init_da3.ply
done
python3 - <<'PY'
import json, shutil
from pathlib import Path
RB=Path('/home/paperspace/data/citrus_all/05_13D_Jackal/blocks_ns/lio_row')
src=RB/'block_005'; dst=RB/'block_005_L239'; dst.mkdir(exist_ok=True)
shutil.copy(src/'transforms.json', dst/'transforms.json')
if (src/'init_da3.ply').exists(): shutil.copy(src/'init_da3.ply', dst/'init_da3.ply')
for V in ['block_005_L239','block_005_L100','block_005_L075','block_005_L050']:
    p=RB/V/'transforms.json'; d=json.loads(p.read_text())
    for f in d['frames']: f.pop('mask_path',None)
    p.write_text(json.dumps(d,indent=2))
    print(V, len(d['frames']), 'frames, ply',(RB/V/'init_da3.ply').exists())
PY
run () {
  local BD=$1 NAME=$2 T0=$(date +%s)
  cd /home/paperspace/code/nerf_new
  echo "n" | MAX_JOBS=4 pixi run ns-train high \
    --data $BD --output-dir $BD/splat_runs_BASE --experiment-name $NAME \
    --pipeline.model.enable-high-features False \
    --pipeline.model.high-loss-weight 0.0 \
    --pipeline.datamanager.semantic-dir $EMPTY \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.stop-split-at 6000 \
    --max-num-iterations 15001 --steps-per-save 14998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "BASE-FAIL $NAME"; return 1; }
  echo "BASE-DONE $NAME in $(( ($(date +%s)-T0)/60 )) min"
}
for v in L100 L239 L075 L050; do run $RB/block_005_$v citrus05_b005_$v; done
echo "=== BASELINE-05-ALL-DONE $(date) ==="
