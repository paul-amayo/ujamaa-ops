#!/bin/bash
# run_geom.sh (A100 session, 11 Nov 2026): the geometry-only step (azalai_rr_geom.py) for the short-gap rungs S1 / S2 of
# RR_addendum2_short_gap.md and the descriptive December comparators of the A100 implementation note (d6c380b), each followed by
# its independent recount (azalai_rr_geom_verify.py --deep). CPU only, one at a time. Writes jobs/DONE_geom.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new
for spec in "apr1130 apr1400" "apr1130 apr16_1230" "dec apr1130" "dec apr1400" "dec apr16_1230"; do
  set -- $spec; T=$1; R=$2; S=_${T}_${R}; [ $T = dec ] && S=${S}_allmnn
  O=$B/geom/${T}_${R}$([ $T = dec ] && echo _allmnn); mkdir -p $O
  $PY $B/bin/azalai_rr_geom.py --twin $T --rev $R --out $O --suffix $S --sitewide > $O/geom.log 2>&1; rc=$?
  $PY $B/bin/azalai_rr_geom_verify.py $O --suffix $S --deep > $O/verify.log 2>&1; rv=$?
  echo "$T $R geom rc=$rc verify rc=$rv $(date -u +%FT%TZ)" >> $B/jobs/DONE_geom.partial
done
mv $B/jobs/DONE_geom.partial $B/jobs/DONE_geom
