#!/bin/bash
# BLOCK-LENGTH BASELINE (2026-08-03): how does reconstruction quality depend
# on block length? Splatfacto-equivalent path (baseline_rgb.sh) so each run
# is ~10 min. Same recipe, same eval protocol, one variable: frame count.
#   L124 31.9 m (as-built, 33% over the documented ~24 m rule)
#   L095 23.9 m (the rule)   L073 18.3 m   L049 12.2 m
# Scoring afterwards: fixed-frame-set renders, foreground-free (no masks
# here) full-image PSNR on train AND eval frames -> the train/eval gap.
set -x
RB=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F
until grep -qa "BASE-ALL-DONE base_L095" /home/paperspace/logs/baseline_L095.log 2>/dev/null; do sleep 30; done
while ps -eo cmd | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 30; done
for V in L124 L073 L049; do
  /home/paperspace/code/automation/baseline_rgb.sh $RB/block_001_$V base_$V 15000 \
    || echo "SWEEP-FAIL $V"
done
echo "=== LENGTH-SWEEP-DONE $(date) ==="
