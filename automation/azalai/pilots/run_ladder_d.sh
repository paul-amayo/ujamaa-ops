#!/bin/bash
# run_ladder_d.sh — December twin, the D rungs: prep (geometry only) and renders for each revisit pass, one after the other.
# Waits for the pair of record's renders first (gates on its DONE files). No outcome is computed here.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; cd /home/paperspace/code/nerf_new
while [ ! -f $B/dec_apr1130/renders/DONEa ] || [ ! -f $B/dec_apr1130/renders/DONEb ]; do sleep 20; done
for REV in apr1054 apr1400 apr16_1230 feb1003 feb1224; do
  R=$B/dec_$REV; mkdir -p $R
  [ -f $R/prep.json ] || $PY $B/bin/azalai_rr_prep.py --twin dec --rev $REV --out $R --registry /mnt/shared/azalai_h3/registry_v0.json --h3prep /mnt/shared/azalai_h3/prep.json > $R/prep.log 2>&1
  [ -f $R/prep.json ] || { echo "prep failed" > $R/FAILED; continue; }
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane1,lane2,lane3,lane4,lane5 --tag a > $R/render_a.log 2>&1 &
  PA=$!
  $PY $B/bin/azalai_rr_render.py $R/prep.json $R/renders --cells lane6,lane7,lane8,lane9,lane10 --tag b > $R/render_b.log 2>&1 &
  PB=$!
  wait $PA $PB
done
date > $B/LADDER_D_DONE
