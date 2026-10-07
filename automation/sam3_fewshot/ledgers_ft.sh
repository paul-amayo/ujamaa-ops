#!/bin/bash
# Fruit ledgers with the few-shot fine-tuned SAM3 (Paul 2026-10-07: "redo relevancy on 04/05 including fruit"): fruit_chain.sh step 1b
# per prod block, same in-tree recipe, --ckpt <trainer checkpoint>, written to <survey>/experimental/sam3_fruit_<tag>/clip_NNN
# (prod/bateleur/sam3_fruit stays the stock ledger). usage: ledgers_ft.sh <survey id> <tag> <ckpt>
set -u; SV=$1; TAG=$2; CK=$3; D=/home/paperspace/data/citrus_all/$SV; OUTD=$D/experimental/sam3_fruit_$TAG; mkdir -p $OUTD
ARU=/home/paperspace/code/aru_sil_core/src; SCR=$ARU/scripts; PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python; L=/home/paperspace/logs/ledgers_${SV}_$TAG.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); say "=== $SV fruit ledgers with $CK -> $OUTD"
for BD in $D/prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]; do N=$(basename $BD | cut -d_ -f2); TID=$BD/supervision/trees_only
  ls $TID/kf_*.png > /dev/null 2>&1 || { say "b$N skip: no tree id maps"; continue; }
  [ -f $OUTD/clip_$N/frame_entries.json ] && { say "b$N done already"; continue; }
  TMP=experimental/sam3_fruit_$TAG/.stage_b$N; rm -rf $D/$TMP
  (cd $ARU && $PY $SCR/fruit_in_trees_ledger.py --data-dir $D --block-dir $BD --out-name $TMP --tree-idmap-dir $TID --ckpt $CK) > /home/paperspace/logs/ledger_${SV}_${TAG}_b$N.log 2>&1 || { say "b$N FAILED"; continue; }
  [ -d $D/$TMP/clip_000 ] && { mv $D/$TMP/clip_000 $OUTD/clip_$N; rmdir $D/$TMP 2>/dev/null; }
  say "b$N $(grep -a '^\[fruit-tree\] [0-9]* instances' /home/paperspace/logs/ledger_${SV}_${TAG}_b$N.log | tail -1 | cut -c1-90)"
done; say "=== $SV ledgers done in $(( ($(date +%s)-t0)/60 )) min"
