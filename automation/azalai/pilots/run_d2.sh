#!/bin/bash
# run_d2.sh (A100 session, 11 Nov 2026): D2 rungs apr1400 and apr16_1230 with the December twin: prep (geometry only) and the renders
# (canon + bag-level twin), one pass after the other; waits (bounded, 2 h) for >= 12 GB free GPU before each pass's renders.
# No outcome is computed here. Writes jobs/DONE_d2_<pass> per pass and jobs/DONE_d2 at the end.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /home/paperspace/code/nerf_new
for REV in apr1400 apr16_1230; do
  R=$B/dec_$REV; mkdir -p $R
  [ -f $R/prep.json ] || $PY $B/bin/azalai_rr_prep.py --twin dec --rev $REV --out $R --registry /mnt/shared/azalai_h3/registry_v0.json --h3prep /mnt/shared/azalai_h3/prep.json > $R/prep.log 2>&1
  [ -f $R/prep.json ] || { echo "prep failed $(date -u +%FT%TZ)" > $B/jobs/DONE_d2_$REV; continue; }
  t0=$(date +%s)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $((tot - used)) -ge 12288 ] && break
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "GPU never had 12 GB free in 2 h $(date -u +%FT%TZ)" > $B/jobs/DONE_d2_$REV; continue 2; }
    sleep 60
  done
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane1,lane2,lane3,lane4,lane5 --tag a > $R/render_a.log 2>&1 &
  PA=$!
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane6,lane7,lane8,lane9,lane10 --tag b > $R/render_b.log 2>&1 &
  PB=$!
  echo "$PA $PB" > $R/render_pids
  wait $PA; ra=$?; wait $PB; rb=$?
  echo "renders rc a=$ra b=$rb DONEa=$([ -f $R/renders/DONEa ] && echo y || echo n) DONEb=$([ -f $R/renders/DONEb ] && echo y || echo n) $(date -u +%FT%TZ)" > $B/jobs/DONE_d2_$REV
done
date -u +%FT%TZ > $B/jobs/DONE_d2
