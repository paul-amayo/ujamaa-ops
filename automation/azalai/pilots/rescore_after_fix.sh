#!/bin/bash
# rescore_after_fix.sh (A100 session, 20:37 UTC): after the empty-mask fix to azalai_rr_score.py (line 48), (1) score + verify the
# apr1400 D2 rung (its first scoring crashed on a stop frame with no SAM 3 mask); (2) re-score the pair of record with the fixed
# scorer under a check suffix, to compare byte for byte with the records of record. Writes jobs/DONE_rescore.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new; R=$B/dec_apr1400
$PY $B/bin/azalai_rr_score.py $R/prep.json $R/renders $B/records --suffix _dec_apr1400 > $R/score.log 2>&1; rs=$?
$PY $B/bin/azalai_rr_verify.py $B/records --suffix _dec_apr1400 --deep 40 --prep $R/prep.json --renders $R/renders > $R/verify.log 2>&1; rv=$?
echo "apr1400 score rc=$rs verify rc=$rv $(date -u +%FT%TZ)" > $B/jobs/DONE_rescore.partial
mkdir -p $B/records_check; R=$B/dec_apr1130
$PY $B/bin/azalai_rr_score.py $R/prep.json $R/renders $B/records_check --suffix _dec_apr1130 > $B/records_check/score_dec_apr1130.log 2>&1; rc=$?
c1=$(cmp -s $B/records_check/rr_targets_dec_apr1130.jsonl $B/records/rr_targets_dec_apr1130.jsonl && echo identical || echo DIFFERENT)
c2=$(cmp -s $B/records_check/rr_summary_dec_apr1130.json $B/records/rr_summary_dec_apr1130.json && echo identical || echo DIFFERENT)
echo "pair of record rescored rc=$rc targets $c1 summary $c2 $(date -u +%FT%TZ)" >> $B/jobs/DONE_rescore.partial
mv $B/jobs/DONE_rescore.partial $B/jobs/DONE_rescore
