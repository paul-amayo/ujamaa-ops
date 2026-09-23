#!/bin/bash
# Chain phase 2 behind the stage-1 fleet for the remaining surveys, in fleet order. For each
# survey: wait until its canonical blocks all have a trained glref stage-1 (tsv rows with
# wall>60 s), then run the per-survey phase-2 driver (which pauses the fleet, seeds every
# block, and resumes the fleet). 05 is handled by citrus05_phase2.sh already running.
TSV=/home/paperspace/logs/citrus_fleet_glref.tsv; LOG=/home/paperspace/logs/citrus_phase2_chain.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
for SV in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal; do
  B=/home/paperspace/data/citrus_all/$SV/prod/tassili/blocks_ns/lio_row100
  NB=$(ls -d $B/block_[0-9][0-9][0-9] | wc -l)
  say "waiting for $SV stage-1: need $NB trained blocks"
  until [ "$(awk -F'\t' -v s=$SV '$1==s && $4>60' $TSV | wc -l)" -ge "$NB" ]; do sleep 600; done
  say "$SV stage-1 complete — starting phase 2"
  bash /home/paperspace/logs/citrus_phase2_survey.sh $SV >> $LOG 2>&1
  say "$SV phase 2 finished: $(tail -n +2 /home/paperspace/logs/citrus_phase2_$SV.tsv | awk -F'\t' '{if($3=="ok"||$3=="skip")o++; else f++} END {printf "ok %d fail %d", o, f}')"
done
say "PHASE-2 CHAIN DONE (01-04)"
