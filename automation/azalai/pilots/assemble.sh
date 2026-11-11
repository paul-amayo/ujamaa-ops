#!/bin/bash
# assemble.sh (A100 session): copy records, results pictures and code into the overnight test folder (no commit here).
B=/mnt/shared/azalai_rr; T=/mnt/shared/overnight_20261112/AZ1_row_to_relevancy; mkdir -p $T/records $T/results $T/code
for f in $B/records/rr_targets_dec_*.jsonl $B/records/rr_summary_dec_*.json $B/records/rr_odo.json $B/records/rr_posthoc_dec_apr1130.json $B/records/rr_rowend_posthoc.json $B/records/rr_posthoc_masks.json; do [ -f $f ] && cp $f $T/records/; done
for d in $B/dec_*; do n=$(basename $d); [ -f $d/rr_prep_summary.json ] && cp $d/rr_prep_summary.json $T/records/rr_prep_summary_$n.json; done
for d in apr1130_apr1400 apr1130_apr16_1230 dec_apr1130_allmnn dec_apr1400_allmnn dec_apr16_1230_allmnn; do cp $B/geom/$d/rr_geom_$d.json $B/geom/$d/rr_geom_targets_$d.jsonl $T/records/; done
cp $B/panels/dec_apr1130/rr_example_dec_apr1130.json $T/records/; cp $B/panels/dec_apr1130/rr_example_dec_apr1130.png $B/panels/dec_apr1130/rr_lane_frames_dec_apr1130.png $T/results/
for f in $B/panels/geom/row_ends_*.png; do cp $f $T/results/; done
cp $B/bin/azalai_rr_*.py $T/code/; cp $B/jobs/*.sh $T/code/
ls -la $T/records $T/results | head -60; du -sh $T/records $T/results $T/code
