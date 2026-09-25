#!/bin/bash
# Full held-out ns-eval (every 10th frame, full resolution) for every trained image_farm segment -> ~/logs/nseval_<seg>.json
# and a summary TSV. The fleet's logged "held-out" figure averaged only 3 single-image samples.
set -uo pipefail
cd /home/paperspace/code/nerf_new; OUT=/home/paperspace/logs/image_farm_nseval.tsv; echo -e "segment\tframes\tpsnr\tssim\tlpips" > $OUT
for sd in $(ls -d /home/paperspace/data/image_farm/IMG_*_s[0-9]); do
  sn=$(basename $sd); CFG=$(ls $sd/blocks_ns/*/block_000/splat_runs_high/*/*/*/config.yml 2>/dev/null | tail -1); [ -n "$CFG" ] || continue
  [ -s /home/paperspace/logs/nseval_$sn.json ] || env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run ns-eval --load-config "$CFG" --output-path /home/paperspace/logs/nseval_$sn.json > /home/paperspace/logs/nseval_$sn.log 2>&1
  python3 -c "
import json; r=json.load(open('/home/paperspace/logs/nseval_$sn.json'))['results']; m=json.load(open('$sd/capture_meta.json'))
print('$sn\t%d\t%.2f\t%.3f\t%.3f' % (m['kept'], r['psnr'], r['ssim'], r['lpips']))" >> $OUT 2>/dev/null || echo -e "$sn\t?\tFAILED" >> $OUT
done
echo "NSEVAL SWEEP DONE"
