#!/bin/bash
# run_extra.sh (A100 session, 11 Nov 2026): the OPTIONAL rendered rungs of the A100 implementation note (d6c380b), after the D2
# renders are done (waits on jobs/DONE_d2, bounded 3 h, gives up if PID 1493630 is gone without it): (1) the pair of record over
# every mutual nearest neighbour pair (sensitivity, not a rule), then the "truth too loose to read" rungs apr1054, feb1003, feb1224
# (counts only, no verdict). Each: prep, renders (GPU, >= 12 GB free, bounded 2 h wait), score, verify --deep 40.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new; t0=$(date +%s)
while [ ! -f $B/jobs/DONE_d2 ]; do
  kill -0 1493630 2>/dev/null || { sleep 10; break; }
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && break
  sleep 30
done
for spec in "apr1130 dec_apr1130_allmnn" "apr1054 dec_apr1054" "feb1003 dec_feb1003" "feb1224 dec_feb1224"; do
  set -- $spec; REV=$1; R=$B/$2; S=_$2; mkdir -p $R
  [ -f $R/prep.json ] || $PY $B/bin/azalai_rr_prep.py --twin dec --rev $REV --out $R --registry /mnt/shared/azalai_h3/registry_v0.json --h3prep /mnt/shared/azalai_h3/prep.json > $R/prep.log 2>&1
  [ -f $R/prep.json ] || { echo "$2 prep failed $(date -u +%FT%TZ)" >> $B/jobs/DONE_extra.partial; continue; }
  t1=$(date +%s); ok=1
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $((tot - used)) -ge 12288 ] && break
    [ $(( $(date +%s) - t1 )) -gt 7200 ] && { ok=0; break; }
    sleep 60
  done
  [ $ok = 1 ] || { echo "$2 GPU never had 12 GB free in 2 h $(date -u +%FT%TZ)" >> $B/jobs/DONE_extra.partial; continue; }
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane1,lane2,lane3,lane4,lane5 --tag a > $R/render_a.log 2>&1 &
  PA=$!
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane6,lane7,lane8,lane9,lane10 --tag b > $R/render_b.log 2>&1 &
  PB=$!
  echo "$PA $PB" > $R/render_pids; wait $PA; wait $PB
  if [ -f $R/renders/DONEa ] && [ -f $R/renders/DONEb ]; then
    $PY $B/bin/azalai_rr_score.py $R/prep.json $R/renders $B/records --suffix $S > $R/score.log 2>&1; rs=$?
    $PY $B/bin/azalai_rr_verify.py $B/records --suffix $S --deep 40 --prep $R/prep.json --renders $R/renders > $R/verify.log 2>&1; rv=$?
    echo "$2 score rc=$rs verify rc=$rv $(date -u +%FT%TZ)" >> $B/jobs/DONE_extra.partial
  else echo "$2 renders incomplete $(date -u +%FT%TZ)" >> $B/jobs/DONE_extra.partial; fi
done
mv $B/jobs/DONE_extra.partial $B/jobs/DONE_extra
