#!/bin/bash
# LiDAR init for every ten_rows row block (CPU). Rig extrinsic via the lidar_init_per_block l2c patch.
ROOT=/home/paperspace/data/klapmuts/dec_2025_ten_rows
LOG=/home/paperspace/logs/tenrows_init_all.log; : > $LOG
cd /home/paperspace/code/aru_sil_core/src/scripts
for BD in /home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100/block_0??; do
  t0=$(date +%s)
  pixi run --manifest-path /home/paperspace/code/InstantSplat/pixi.toml python lidar_init_per_block.py --block-dir "$BD" --root "$ROOT" --force >> $LOG 2>&1 \
    && echo "INIT OK $(basename $BD) $(( $(date +%s) - t0 ))s $(stat -c %s $BD/init_lidar.ply 2>/dev/null) B" | tee -a $LOG \
    || { echo "INIT FAIL $(basename $BD)" | tee -a $LOG; grep -vE "^I2026|WARN|│|╭|╰|·" $LOG | tail -3; }
done
echo "INIT-ALL DONE $(date '+%H:%M:%S')" | tee -a $LOG
