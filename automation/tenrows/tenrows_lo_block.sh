#!/bin/bash
# tenrows_lo_block.sh <NNN> — one ten_rows block on LiDAR-odometry poses: block_NNN_lo/ (transforms.json = the block's
# from-scratch SfM Sim(3)-aligned onto the KISS-ICP trajectory, LiDAR init lifted from the raw scans with the LO poses),
# trained with the stage-1 recipe of the refined fleet (ns-train high, RGB-only, antialiased, 15k iters, every-10th
# keyframe held out), then ns-eval (PSNR) with the held-out renders saved for the qualitative strip.
set -u
N=${1:?block number, e.g. 020}; SUF=${2:-_lo}; R=/home/paperspace/data/klapmuts/dec_2025_ten_rows; B=$R/prod/tassili/blocks_ns/lio_row100; BD=$B/block_${N}${SUF}; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/tenrows_lo_block_${N}${SUF}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
unset CUDA_HOME LD_LIBRARY_PATH   # the nerf_new pixi env brings its own CUDA runtime; the glomap CUDA-12 tree must not leak in (libcudart.so.12 import failure, 05:20)
[ -e $BD/init_lidar.ply ] || { say "no LO LiDAR init in $BD yet (tenrows_lo_lidar_init.py)"; exit 1; }
say "=== block_${N}${SUF}: $(python3 -c "import json;t=json.load(open('$BD/transforms.json'));print(len(t['frames']),'frames, ply',t.get('ply_file_path'))")"
rm -rf $BD/splat_runs_STAGE1; t0=$(date +%s)
( cd $NS && echo "n" | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1" --experiment-name stage1 \
    --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
    --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 6000 --pipeline.model.sky-loss-lambda 0.0 --pipeline.model.report-masked-metrics False \
    --max-num-iterations 15001 --steps-per-save 5000 --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 ) > /home/paperspace/logs/tenrows_lo_stage1_${N}.log 2>&1
rc=$?; say "train rc=$rc in $(( $(date +%s)-t0 ))s"
CFG=$(ls -d $BD/splat_runs_STAGE1/stage1/high/*/config.yml 2>/dev/null | tail -1); [ -n "$CFG" ] || { say "no config.yml"; exit 1; }
t0=$(date +%s); ( cd $NS && pixi run ns-eval --load-config "$CFG" --output-path $BD/eval_lo.json --render-output-path $BD/eval_renders ) > /home/paperspace/logs/tenrows_lo_nseval_${N}.log 2>&1
say "ns-eval rc=$? in $(( $(date +%s)-t0 ))s: $(python3 -c "import json;r=json.load(open('$BD/eval_lo.json'))['results'];print('PSNR %.2f  SSIM %.3f  LPIPS %.3f' % (r['psnr'], r['ssim'], r['lpips']))" 2>/dev/null)"
say "=== block_${N}${SUF} DONE"
