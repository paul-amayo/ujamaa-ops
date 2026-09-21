#!/bin/bash
# Containment verdict sweep over one survey's glref stage-2 seeds: per block, the recipe's
# top frame (most painted supervision pixels) -> containment_eval.py -> lines prefixed the way
# distill_containment_verdicts.py expects -> verdicts_censusinit_glref.json in the cfg dir.
SV=${1:?survey}; S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/tassili/blocks_ns/lio_row100
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json
KF=$S/prod/scratch_sam3; LOG=/home/paperspace/logs/containment_sweep_${SV}_glref.log; : > $LOG
FIG=/home/paperspace/logs/contain_figs_$SV; mkdir -p $FIG
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a /home/paperspace/logs/verdict_sweep_${SV}.log; }
cd /home/paperspace/code/nerf_new
say "SWEEP START $SV"
for BD in $B/block_[0-9][0-9][0-9]; do
  N=$(basename $BD); NN=${N#block_}; SUP=$BD/supervision/trees_only
  CFG=$(ls -t $BD/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml 2>/dev/null | head -1)
  [ -n "$CFG" ] && [ -d "$SUP" ] || { say "$N: no glref seed or supervision — skipped"; continue; }
  FR=$(pixi run python - "$SUP" << 'PY'
import sys, numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16); n = int((a != 65535).sum())
    if n > best[0]: best = (n, f.name)
print(best[1])
PY
)
  [ -n "$FR" ] && [ "$FR" != "None" ] || { say "$N: no painted frame — skipped"; continue; }
  t0=$(date +%s)
  HIGH_EMBEDDER_CKPT=$EMB timeout 900 pixi run python /home/paperspace/code/aru_sil_core/src/scripts/containment_eval.py \
    --config $CFG --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF \
    --out $FIG/${N}_$FR 2>&1 | grep -aE "^(TREE|ROW) " | sed "s/^/[$NN $FR] /" >> $LOG
  say "$N $FR: $(grep -c "^\[$NN " $LOG) lines, $(( $(date +%s)-t0 ))s"
done
say "SWEEP DONE — distilling"
python3 /home/paperspace/code/automation/distill_containment_verdicts.py --log $LOG --out $B/verdicts_censusinit_glref.json $DISTILL_EXTRA 2>&1 | tail -5 | tee -a /home/paperspace/logs/verdict_sweep_${SV}.log
say "VERDICTS WRITTEN: $B/verdicts_censusinit_glref.json"
