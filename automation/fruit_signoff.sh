#!/bin/bash
# FRUIT SIGN-OFF battery (TESTING.md §4b) — run ALL documented tests on a
# fruit-bearing checkpoint and emit one report. Partial runs cannot sign off.
#   usage: fruit_signoff.sh <run_dir> [report_path]
# Assumes citrus 04 block_001 family (the fruit scene). GPU required.
set -uo pipefail
RD=$(readlink -f "$1")
REPORT=${2:-/home/paperspace/logs/fruit_signoff_$(basename $RD).txt}
B=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F
H=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
HJ=/home/paperspace/data/citrus_all/04_13D_Jackal/scene_graph_v4/marker_hierarchy_fruit5.json
SUP=$B/block_001/supervision/strict_tree_v2
CFG=$(ls -t $RD/high/*/config.yml | head -1)
FIG=/home/paperspace/code/lab_notebook/figs
TAG=$(basename $RD)
cd /home/paperspace/code/nerf_new
{
echo "=============================================================="
echo "FRUIT SIGN-OFF — $TAG — $(date)"
echo "run: $RD"
echo "config: $CFG"
echo "=============================================================="

echo; echo "--- [1] APPEARANCE (score_splat + render) ---"
pixi run python /home/paperspace/code/automation/score_splat.py "$RD" 2>/dev/null \
  | grep -aE "^[a-z_0-9]+ +|train|eval" | head -3
pixi run python /home/paperspace/logs/render_one_base.py "$RD" kf_000542.png \
  $FIG/signoff_${TAG}_542.png /home/paperspace/data/citrus_all/04_13D_Jackal/kf_images 2>/dev/null | tail -1

echo; echo "--- [2+3+4] POINTING MAPS (no-walk = headline, walked, anatomy) ---"
for MODE in "--no-walk" ""; do
  HIGH_EMBEDDER_CKPT=$H pixi run python \
    /home/paperspace/code/aru_sil_core/src/scripts/fruit_pointing_map.py \
    --config $CFG --hyper-ckpt $H --hierarchy-json $HJ --supervision-dir $SUP \
    --frame kf_000542.png $MODE \
    --out $FIG/signoff_${TAG}_pointing$( [ -n "$MODE" ] && echo _nowalk ).png \
    2>/dev/null | grep -aE "CROSS-LEVEL|FP anatomy"
done

echo; echo "--- [5] CEILING CONTROL (must stay 100/100/0) ---"
HIGH_EMBEDDER_CKPT=$H pixi run python \
  /home/paperspace/code/aru_sil_core/src/scripts/fruit_pointing_map.py \
  --config $CFG --hyper-ckpt $H --hierarchy-json $HJ --supervision-dir $SUP \
  --frame kf_000542.png --gt-features --out /tmp/signoff_ceiling.png \
  2>/dev/null | grep -aE "CROSS-LEVEL"

echo; echo "--- [6] FULL RELEVANCY EVAL (multi-frame, negatives-free) ---"
HIGH_EMBEDDER_CKPT=$H pixi run python \
  /home/paperspace/code/aru_sil_core/src/scripts/eval_r6_relevancy.py \
  --config $CFG --hyper-ckpt $H --hierarchy-json $HJ --block-dir $B/block_001 \
  --markers 1,16,24,60,73 --n-frames 12 --no-negatives \
  --out-dir /tmp/signoff_r6 2>/dev/null \
  | grep -aE "POINTING|FRUIT-PIXEL|MEAN IoU|fruit@"

echo; echo "--- [7] ALIGNED GATES (own supervision) ---"
if [ -f /home/paperspace/logs/aligned_gates.py ]; then
  pixi run python /home/paperspace/logs/aligned_gates.py "$CFG" 2>/dev/null | tail -6 \
    || echo "aligned_gates run failed — mark PARTIAL"
else
  echo "aligned_gates.py missing — mark PARTIAL"
fi

echo; echo "--- [8] CROSS-VIEW CONSISTENCY (542 -> 543) ---"
if [ -f /home/paperspace/logs/s9b_542_to_543.py ]; then
  pixi run python /home/paperspace/logs/s9b_542_to_543.py "$CFG" 2>/dev/null | tail -4 \
    || echo "cross-view script needs config wiring — mark PARTIAL"
else
  echo "s9b_542_to_543.py missing — mark PARTIAL"
fi

echo; echo "=============================================================="
echo "SIGN-OFF: [ ] Paul approved (record in notebook)   [ ] PARTIAL"
echo "=============================================================="
} 2>&1 | tee "$REPORT"
echo "report -> $REPORT"
