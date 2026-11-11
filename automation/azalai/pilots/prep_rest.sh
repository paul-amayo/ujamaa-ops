#!/bin/bash
cd /home/paperspace/code/nerf_new
D=/mnt/shared/azalai_online
for job in "apr1054 graph" "apr1054 lo" "apr1400 lo" "apr1400 graph" "apr16_1230 lo" "apr16_1230 graph"; do
  set -- $job
  /home/paperspace/.pixi/bin/pixi run python $D/bin/azalai_online_prep.py --rev $1 --source $2 --out $D/causal > $D/logs/prep_$1_$2.log 2>&1
done
touch $D/causal/PREP_REST_DONE
