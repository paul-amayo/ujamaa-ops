#!/bin/bash
# A100 agent, 11 Nov 2026: azalai_online_verify.py (unchanged, as received) on every run not yet verified
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /home/paperspace/code/nerf_new
D=/mnt/shared/azalai_online
PY="/home/paperspace/.pixi/bin/pixi run python $D/bin/azalai_online_verify.py --records $D/records --causal $D/causal"
for job in "apr1130 apr1400" "apr1130 apr16_1230" "dec apr1400" "dec apr16_1230" "dec apr1054" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 > $D/logs/verify_$1_$2.log 2>&1; echo "$1 $2 rc=$?" >> $D/logs/verify_rest.rc
done
for job in "dec apr1054" "apr1054 apr1130" "apr1054 apr1400" "apr1054 apr16_1230"; do
  set -- $job
  $PY --reg $1 --rev $2 --direct > $D/logs/verify_$1_$2_direct.log 2>&1; echo "$1 $2 direct rc=$?" >> $D/logs/verify_rest.rc
done
touch $D/logs/VERIFY_REST_DONE
