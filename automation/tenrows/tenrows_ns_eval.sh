#!/bin/bash
# All-held-out-images eval (PSNR/SSIM/LPIPS) for finished ten_rows stage-1 runs.
# usage: tenrows_ns_eval.sh <block_dir> [<block_dir> ...]   -> <block_dir>/stage1_eval.json
NS=/home/paperspace/code/nerf_new
for BD in "$@"; do
  CFG=$(ls -t $BD/splat_runs_STAGE1/stage1/high/*/config.yml 2>/dev/null | head -1)
  [ -z "$CFG" ] && { echo "[ns-eval] $(basename $BD): no config.yml"; continue; }
  NAME=$(basename $BD); t0=$(date +%s)
  ( cd $NS && pixi run ns-eval --load-config "$CFG" --output-path "$BD/stage1_eval.json" ) > /home/paperspace/logs/tenrows_nseval_$NAME.log 2>&1
  python3 -c "
import json; m = json.load(open('$BD/stage1_eval.json'))['results']
print('[ns-eval] $NAME: psnr %.2f  ssim %.4f  lpips %.4f  (%d s)' % (m['psnr'], m['ssim'], m['lpips'], $(date +%s) - $t0))" 2>/dev/null || { echo "[ns-eval] $NAME FAILED"; tail -3 /home/paperspace/logs/tenrows_nseval_$NAME.log; }
done
