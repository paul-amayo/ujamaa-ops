#!/bin/bash
# demo_rowpick.sh — Paul, 2026-09-27: "why does pine tree matter, we ask a row query so the flipping is irrelevant".
# Right: the row query is answered by the row field alone. The crossing was not the field flipping — it was the compositor
# comparing the two row words as (score - that word's own per-block verdict threshold), so block 023's oak threshold of
# 0.70 (IoU 0.463, prec 0.464 — fitted on a frame that is nearly all pine row) beat pine's 0.85 on pixels the field scores
# oak 0.762 / pine 0.871. Fix is at row level: pick the row word the field scores highest (--row-pick raw); thresholds only
# gate whether ANY row is claimed. No tree identity, no marker_hierarchy.json in the answer path.
# Re-renders the maps (both picks stored per pixel), composites with the raw pick, and scores the crossing both ways.
set -uo pipefail
SV=05_13D_Jackal; D=/home/paperspace/logs/demo_05; S=/home/paperspace/data/citrus_all/$SV
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
BLOCKS="018 019 020 021 022 023"; L=/home/paperspace/logs/demo_rowpick.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -e $D/demo_v3_rowfromtree.mp4 ] || cp $D/demo.mp4 $D/demo_v3_rowfromtree.mp4
cd /home/paperspace/code/nerf_new
say "=== pass 1+2: maps carry both row picks; composite with --row-pick raw"
HIGH_EMBEDDER_CKPT=$EMB pixi run python /home/paperspace/logs/sidecar_demo_overlay.py --survey $SV --path $D/demo_path.json \
  --models $BLOCKS --backdrop-dir $D/backdrop --out $D --row-pick raw >> $L 2>&1 || { say "overlay failed"; exit 1; }
mv $D/demo.mp4 $D/demo_v4_rowpick_raw.mp4
say "=== crossing, scored on the same maps both ways (A = row decode disagrees with the hierarchy row of the tree there)"
for PICK in margin raw; do
  mkdir -p $D/rowcross_$PICK
  pixi run python /home/paperspace/logs/sidecar_row_cross_diag.py --survey $SV --path $D/demo_path.json --maps $D/maps \
    --blocks $BLOCKS --backdrop-dir $D/backdrop --out $D/rowcross_$PICK --row-pick $PICK --top 4 2>&1 | grep -a '^\[cross\]' | sed "s/^/[$PICK] /" | tee -a $L
done
say "done: $D/demo_v4_rowpick_raw.mp4 ($(du -h $D/demo_v4_rowpick_raw.mp4 | cut -f1))"
