#!/bin/bash
# Citrus lio_row100 integrity fleet, phase 1: per survey, (a) GL-flip + GLOMAP-refine every
# canonical block, then (b) stage-1 retrain (prod flags, experiment stage1_bg00_glref).
# Resumable: flip/refine are tag-guarded no-ops when done; stage-1 skipped if the glref run exists.
# Old runs untouched (prod doctrine: never delete). Stage-2/verdicts are phase 2.
SURVEYS="01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal 05_13D_Jackal"
NS=/home/paperspace/code/nerf_new
LOG=/home/paperspace/logs/citrus_fleet_glref.log; TSV=/home/paperspace/logs/citrus_fleet_glref.tsv
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
[ -f $TSV ] || echo -e "survey\tblock\trefine_p50_m\tstage1_wall_s\ttrain_fg\teval_fg\teval_psnr" > $TSV
for SV in $SURVEYS; do
  CFG=/home/paperspace/data/citrus_all/$SV/prod/tassili/blocks_ns/lio_row100
  say "=== $SV: $(ls -d $CFG/block_[0-9][0-9][0-9] | wc -l) canonical blocks ==="
  for BD in $CFG/block_[0-9][0-9][0-9]; do
    B=$(basename $BD)
    python3 /home/paperspace/logs/citrus_flip_block.py $BD >> $LOG 2>&1 || { say "$SV/$B FLIP REFUSED — skipping block"; continue; }
    if ! grep -q "COLMAP" $BD/transforms.json; then
      /home/paperspace/logs/citrus_refine_block.sh $BD >> $LOG 2>&1 || say "$SV/$B refine failed — continuing on GL odometry poses"
    fi
    P50=$(grep -oE "p50 [0-9.]+ m" $BD/transforms.json 2>/dev/null | head -1); P50=${P50:-odo}
    if ls $BD/splat_runs_STAGE1/stage1_bg00_glref/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1 || ls $BD/splat_runs_STAGE1/stage1_bg00_glref/high/*/config.yml >/dev/null 2>&1; then
      say "$SV/$B stage-1 glref exists — skip"; continue
    fi
    ENVS=(MAX_JOBS=4 CANARY_EVERY=2000 TREE_WEIGHT_BG=0.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
    if [ -d $BD/supervision ]; then ENVS+=(TREE_WEIGHT_DIR=$BD/supervision); else say "$SV/$B: no supervision dir (training without TREE_WEIGHT)"; fi
    t0=$(date +%s)
    ( cd $NS && echo "n" | env "${ENVS[@]}" \
      pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1" --experiment-name stage1_bg00_glref \
      --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 \
      --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic \
      --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 6000 \
      --pipeline.model.sky-loss-lambda 1.0 --pipeline.model.report-masked-metrics True \
      --max-num-iterations 15001 --steps-per-save 5000 --vis tensorboard \
      nerfstudio-data --eval-mode interval --eval-interval 10 ) > /home/paperspace/logs/citrus_s1_${SV}_${B}.log 2>&1
    rc=$?; wall=$(( $(date +%s)-t0 ))
    M=$(cd $NS && pixi run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, numpy as np
r = sorted(glob.glob('$BD/splat_runs_STAGE1/stage1_bg00_glref/high/*/'))[-1]
ea = EventAccumulator(r); ea.Reload(); o = []
for tag in ('Train Metrics Dict/psnr_fg', 'Eval Metrics Dict/psnr_fg', 'Eval Images Metrics/psnr'):
    v = [e.value for e in ea.Scalars(tag)] if tag in ea.Tags()['scalars'] else []
    o.append(f'{np.mean(v[-3:]):.2f}' if v else 'NA')
print('\t'.join(o))" 2>/dev/null | tail -1)
    A=$(python3 -c "
import json; t = json.load(open('$BD/transforms.json'))
import re; m = re.search(r'p50 ([0-9.]+)', '') ; print('ref' if 'COLMAP' in t.get('pose_convention','') else 'odo')")
    echo -e "$SV\t$B\t$A\t$wall\t$M" >> $TSV
    say "$SV/$B rc=$rc wall=${wall}s poses=$A fg train/eval + eval: $M"
  done
  say "=== $SV stage-1 fleet done ==="
done
say "CITRUS-GLREF PHASE-1 DONE"
