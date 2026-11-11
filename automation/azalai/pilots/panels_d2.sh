#!/bin/bash
# panels_d2.sh <pass> (A100 session): eye-check sheets (2 per outcome, seed 0) and the plan view for one D2 rung. Writes jobs/DONE_panels_<pass>.
B=/mnt/shared/azalai_rr; PY="/home/paperspace/.pixi/bin/pixi run python"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/paperspace/code/nerf_new; REV=$1; R=$B/dec_$REV
$PY $B/bin/azalai_rr_panel.py $R/prep.json $R/renders $B/records/rr_targets_dec_$REV.jsonl $B/panels/dec_$REV --suffix _dec_$REV --eye 2 --plan > $R/panel.log 2>&1; rc=$?
echo "panels $REV rc=$rc $(date -u +%FT%TZ)" > $B/jobs/DONE_panels_$REV
