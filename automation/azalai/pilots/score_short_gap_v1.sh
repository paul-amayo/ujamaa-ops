#!/bin/bash
# A100 agent, 11 Nov 2026: S1/S2 short-gap rungs of RR_addendum2 (note 5a4467b/585175f), scorer unchanged since f884d15
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /home/paperspace/code/nerf_new
D=/mnt/shared/azalai_online
for rev in apr1400 apr16_1230; do
  /home/paperspace/.pixi/bin/pixi run python $D/bin/azalai_online_score.py --reg apr1130 --rev $rev --causal $D/causal --out $D/records > $D/logs/score_apr1130_$rev.log 2>&1
  echo "apr1130 $rev rc=$?" >> $D/logs/score_short_gap.rc
done
touch $D/records/SCORE_SHORT_GAP_DONE
