#!/bin/bash
# Refined-pose pass: after SfM-all and the odometry-pose fleet, build block_NNN_ref for every block
# (COLMAP poses Sim3-aligned into the LIO world, LiDAR init kept) and train stage-1 on each.
B=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100; T=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili
LOG=/home/paperspace/logs/tenrows_stage1_driver_ref.log; PSNR=/home/paperspace/logs/tenrows_stage1_psnr_ref.tsv; NS=/home/paperspace/code/nerf_new
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
until grep -q "SFM-ALL DONE" /tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/tasks/bpxrgxwqf.output 2>/dev/null; do sleep 60; done
until grep -q "STAGE1-ALL DONE" /home/paperspace/logs/tenrows_stage1_driver.log 2>/dev/null; do sleep 60; done
log "SfM-all + odometry fleet done — building refined blocks"
n=0; for BD in $B/block_0??; do blk=$(basename $BD); [ -f $T/colmap_$blk/transforms.json ] || [ "$blk" = block_013 ] || { log "$blk: no COLMAP model, skipped"; continue; }; python3 /home/paperspace/logs/tenrows_refine_block.py $blk | tee -a $LOG; n=$((n+1)); done; log "refined blocks built: $n"
echo -e "block\tframes\titers\twall_s\ttrain_psnr_last1k\teval_psnr_last\teval_psnr_mean_last" > $PSNR
for BD in $B/block_0??_ref; do
  NAME=$(basename $BD); [ "$NAME" = block_013_ref ] && { log "block_013_ref already trained (17.52 dB)"; continue; }
  rm -rf $BD/splat_runs_STAGE1; t0=$(date +%s)
  ( cd $NS && echo "n" | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1" --experiment-name stage1 \
      --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
      --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 6000 --pipeline.model.sky-loss-lambda 0.0 --pipeline.model.report-masked-metrics False \
      --max-num-iterations 15001 --steps-per-save 5000 --vis tensorboard nerfstudio-data --eval-mode interval --eval-interval 10 ) > /home/paperspace/logs/tenrows_stage1_$NAME.log 2>&1
  rc=$?; wall=$(( $(date +%s) - t0 )); RUN=$(ls -d $BD/splat_runs_STAGE1/stage1/high/*/ 2>/dev/null | tail -1)
  M=$(cd $NS && pixi run python -c "
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$RUN'); ea.Reload(); out = []
for tag in ['Train Metrics Dict/psnr', 'Eval Images Metrics/psnr']:
    if tag in ea.Tags().get('scalars', []):
        v = np.array([e.value for e in ea.Scalars(tag)]); s = np.array([e.step for e in ea.Scalars(tag)])
        out += [f'{np.mean(v[s >= s.max()-1000]):.2f}'] if 'Train' in tag else [f'{v[-1]:.2f}', f'{np.mean(v[-3:]):.2f}']
    else: out += ['NA'] if 'Train' in tag else ['NA', 'NA']
print('\t'.join(out))" 2>/dev/null | tail -1)
  nf=$(python3 -c "import json; print(len(json.load(open('$BD/transforms.json'))['frames']))")
  echo -e "$NAME\t$nf\t15001\t$wall\t$M" >> $PSNR; log "$NAME rc=$rc wall=${wall}s train/eval PSNR: $M"
done
log "STAGE1-REF-ALL DONE"; /home/paperspace/logs/tenrows_ns_eval.sh $B/block_0??_ref | tee -a $LOG; log "REF-EVAL DONE"
