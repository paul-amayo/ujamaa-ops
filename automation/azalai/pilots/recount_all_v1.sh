#!/bin/bash
# A100 agent, 11 Nov 2026: own recount (azalai_online_recount.py) on every online run
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /home/paperspace/code/nerf_new
D=/mnt/shared/azalai_online
PY="/home/paperspace/.pixi/bin/pixi run python $D/bin/azalai_online_recount.py --records $D/records --out $D/recount"
for job in "dec apr1130" "apr1130 apr1400" "apr1130 apr16_1230" "dec apr1400" "dec apr16_1230" "dec apr1054" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 > $D/logs/recount_$1_$2.log 2>&1; echo "$1 $2 rc=$?" >> $D/logs/recount_all.rc
done
for job in "dec apr1054" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 --direct > $D/logs/recount_$1_$2_direct.log 2>&1; echo "$1 $2 direct rc=$?" >> $D/logs/recount_all.rc
done
touch $D/logs/RECOUNT_ALL_DONE
