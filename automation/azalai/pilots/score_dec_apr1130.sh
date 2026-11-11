#!/bin/bash
# score + verify the pair of record (A100 session, 11 Nov 2026). Writes DONE_score_dec_apr1130 with the exit codes.
B=/mnt/shared/azalai_rr; export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /home/paperspace/code/nerf_new
/home/paperspace/.pixi/bin/pixi run python $B/bin/azalai_rr_score.py $B/dec_apr1130/prep.json $B/dec_apr1130/renders $B/records --suffix _dec_apr1130 > $B/dec_apr1130/score.log 2>&1
rc=$?
echo "score rc=$rc $(date -u +%FT%TZ)" > $B/jobs/DONE_score_dec_apr1130
