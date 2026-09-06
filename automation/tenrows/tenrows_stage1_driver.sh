#!/bin/bash
# ten_rows stage-1 driver: waits for the QLoRA pause, then (1) the A/B pair on block_013
# (kf arm vs full-frame arm, same held-out keyframes), then (2) stage-1 on every other block.
# Citrus stage-1 flags minus supervision (no TREE_WEIGHT), sky loss OFF (polytunnel roof), no masks.
ROOT=/home/paperspace/data/klapmuts/dec_2025_ten_rows; BLKS=$ROOT/prod/tassili/blocks_ns/lio_row100
LOG=/home/paperspace/logs/tenrows_stage1_driver.log; PSNR=/home/paperspace/logs/tenrows_stage1_psnr.tsv
NS=/home/paperspace/code/nerf_new
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
until grep -q "PAUSE DONE" /home/paperspace/logs/g0_pause.log 2>/dev/null; do sleep 60; done
log "GPU free (QLoRA paused): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo -e "block\tframes_train\tframes_test\titers\twall_s\ttrain_psnr_last1k\teval_psnr_last\teval_psnr_mean_last" > $PSNR
train_one(){  # $1 block dir, $2 eval args
  BD=$1; NAME=$(basename $BD); until [ -s $BD/init_lidar.ply ]; do sleep 30; done
  rm -rf $BD/splat_runs_STAGE1 /home/paperspace/code/nerf_new/outputs/$NAME; t0=$(date +%s)
  ( cd $NS && echo "n" | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1" --experiment-name stage1 \
      --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
      --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 6000 --pipeline.model.sky-loss-lambda 0.0 --pipeline.model.report-masked-metrics False \
      --max-num-iterations 15001 --steps-per-save 5000 --vis tensorboard nerfstudio-data $2 ) > /home/paperspace/logs/tenrows_stage1_$NAME.log 2>&1
  rc=$?; wall=$(( $(date +%s) - t0 ))
  RUN=$(ls -d $BD/splat_runs_STAGE1/stage1/high/*/ 2>/dev/null | tail -1)
  read ntr nte < <(python3 -c "
import json,sys; t=json.load(open('$BD/transforms.json')); print(len(t.get('train_filenames', t['frames'])), len(t.get('test_filenames', [])) or 'interval10')")
  M=$(cd $NS && pixi run python -c "
import numpy as np, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$RUN'); ea.Reload(); out = []
for tag in ['Train Metrics Dict/psnr', 'Eval Images Metrics/psnr']:
    if tag in ea.Tags().get('scalars', []):
        v = np.array([e.value for e in ea.Scalars(tag)]); s = np.array([e.step for e in ea.Scalars(tag)])
        out += [f'{np.mean(v[s >= s.max()-1000]):.2f}', f'{v[-1]:.2f}'] if 'Train' in tag else [f'{v[-1]:.2f}', f'{np.mean(v[-3:]):.2f}']
    else: out += ['NA', 'NA']
print('\t'.join([out[0], out[2], out[3]]))" 2>/dev/null | tail -1)
  echo -e "$NAME\t$ntr\t$nte\t15001\t$wall\t$M" >> $PSNR
  log "$NAME rc=$rc wall=${wall}s train/eval PSNR: $M"
}
log "A/B on block_013: kf arm then full arm (explicit train/test filenames)"
train_one $BLKS/block_013 "--eval-mode filename"
train_one $BLKS/block_013_full "--eval-mode filename"
log "A/B DONE — remaining blocks"
for BD in $BLKS/block_0??; do [ "$(basename $BD)" = block_013 ] && continue; train_one $BD "--eval-mode interval --eval-interval 10"; done
log "STAGE1-ALL DONE"; echo "STAGE1-ALL DONE" >> $LOG
