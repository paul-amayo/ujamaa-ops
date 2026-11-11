#!/bin/bash
cd /home/paperspace/code/nerf_new
D=/mnt/shared/azalai_online
PY="/home/paperspace/.pixi/bin/pixi run python $D/bin/azalai_online_score.py --causal $D/causal --out $D/records"
for job in "dec apr1054" "dec apr1400" "dec apr16_1230" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 > $D/logs/score_$1_$2.log 2>&1
done
for job in "dec apr1054" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 --truth direct > $D/logs/score_$1_$2_direct.log 2>&1
done
touch $D/records/SCORE_REST_DONE
