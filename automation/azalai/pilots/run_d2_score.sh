#!/bin/bash
# run_d2_score.sh (A100 session, 11 Nov 2026): score + verify each D2 rung once its renders are done (waits on jobs/DONE_d2_<pass>,
# bounded 3 h, and gives up if the render driver PID 1493630 is gone without it). Writes jobs/DONE_d2_score.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new; t0=$(date +%s)
for REV in apr1400 apr16_1230; do
  R=$B/dec_$REV
  while [ ! -f $B/jobs/DONE_d2_$REV ]; do
    kill -0 1493630 2>/dev/null || { sleep 10; break; }
    [ $(( $(date +%s) - t0 )) -gt 10800 ] && break
    sleep 30
  done
  if [ -f $R/renders/DONEa ] && [ -f $R/renders/DONEb ]; then
    $PY $B/bin/azalai_rr_score.py $R/prep.json $R/renders $B/records --suffix _dec_$REV > $R/score.log 2>&1; rs=$?
    $PY $B/bin/azalai_rr_verify.py $B/records --suffix _dec_$REV --deep 40 --prep $R/prep.json --renders $R/renders > $R/verify.log 2>&1; rv=$?
    echo "$REV score rc=$rs verify rc=$rv $(date -u +%FT%TZ)" >> $B/jobs/DONE_d2_score.partial
  else
    echo "$REV renders not complete, not scored $(date -u +%FT%TZ)" >> $B/jobs/DONE_d2_score.partial
  fi
done
mv $B/jobs/DONE_d2_score.partial $B/jobs/DONE_d2_score
