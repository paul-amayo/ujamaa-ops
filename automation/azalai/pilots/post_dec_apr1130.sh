#!/bin/bash
# post_dec_apr1130.sh (A100 session): verify (recount + --deep 40 + Part 3) and panels for the pair of record. Writes jobs/DONE_post_dec_apr1130.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new
$PY $B/bin/azalai_rr_verify.py $B/records --suffix _dec_apr1130 --odo $B/records/rr_odo.json --deep 40 --prep $B/dec_apr1130/prep.json --renders $B/dec_apr1130/renders > $B/dec_apr1130/verify.log 2>&1 &
PV=$!
$PY $B/bin/azalai_rr_panel.py $B/dec_apr1130/prep.json $B/dec_apr1130/renders $B/records/rr_targets_dec_apr1130.jsonl $B/panels/dec_apr1130 --suffix _dec_apr1130 --eye 4 --example --plan --part1 4 > $B/dec_apr1130/panel.log 2>&1
rp=$?; wait $PV; rv=$?
echo "verify rc=$rv panel rc=$rp $(date -u +%FT%TZ)" > $B/jobs/DONE_post_dec_apr1130
