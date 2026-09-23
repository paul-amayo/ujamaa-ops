#!/bin/bash
# Phase-2 chain, v2 (09-22). v1 waited on fleet tsv rows and hung on 01 (69 rows, 71 trained). Readiness is now
# measured on ARTIFACTS (every canonical block has a final stage-1 glref checkpoint), and the chain sweeps the
# surveys repeatedly so a survey with blocks still missing (02: five retrains pending) does not block the others.
LOG=/home/paperspace/logs/citrus_phase2_chain2.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
counts(){ local B=/home/paperspace/data/citrus_all/$1/prod/tassili/blocks_ns/lio_row100
  echo "$(ls -d $B/block_[0-9][0-9][0-9] | wc -l) $(ls -d $B/block_[0-9][0-9][0-9]/splat_runs_STAGE1/stage1_bg00_glref/high/*/nerfstudio_models/*.ckpt 2>/dev/null | wc -l)"; }
seeded(){ local B=/home/paperspace/data/citrus_all/$1/prod/tassili/blocks_ns/lio_row100
  ls -d $B/block_[0-9][0-9][0-9]/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/nerfstudio_models/*.ckpt 2>/dev/null | wc -l; }
declare -A DONE
say "chain v2 up: $(for s in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal; do echo -n "$s=$(counts $s | tr ' ' '/') "; done)"
while true; do
  left=0
  for SV in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal; do
    [ "${DONE[$SV]}" = 1 ] && continue
    read NB CK <<< "$(counts $SV)"
    if [ "$CK" -ge "$NB" ]; then
      say "$SV stage-1 complete ($CK/$NB) — phase 2 (seeded so far: $(seeded $SV))"
      bash /home/paperspace/logs/citrus_phase2_survey.sh $SV >> $LOG 2>&1
      say "$SV phase 2 finished: seeded $(seeded $SV)/$NB; $(tail -n +2 /home/paperspace/logs/citrus_phase2_$SV.tsv 2>/dev/null | awk -F'\t' '{if($3=="ok"||$3=="skip")o++; else f++} END {printf "ok %d fail %d", o, f}')"
      DONE[$SV]=1
    else
      left=$((left+1))
    fi
  done
  [ "$left" -eq 0 ] && break
  sleep 900
done
say "PHASE-2 CHAIN v2 DONE (01-04)"
