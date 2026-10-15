#!/bin/bash
# e3_lite_cell.sh — LangSplat-lite on one identity cell (paper E3 fallback, 2026-10-13): SAM3 mask-crop CLIP features (no labels) -> PCA-32 per-pixel
# targets on the census frames -> alpha-weighted lift into the gaussians (no training) -> renders of the verdict frames -> E1 scorer in flat
# space (inverse-PCA decode, cosine to the cell's words; instance words are expected to fail, category words come with the category protocol).
#   N=<ws>/identity [KF=<kf dir>] [FRUIT=1] e3_lite_cell.sh
set -uo pipefail; : ${N:?}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; A=$N/arms; T=$A/lite_targets; mkdir -p $A
CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); WS=$(dirname $N); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); L=/home/paperspace/logs/e3_lite.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(echo $N | awk -F/ '{print $6"/"$8}'): $*" | tee -a $L; }; t0=$(date +%s)
run(){ (cd $NS && nice -n 10 pixi run python "$@" 2>&1) | grep -aE '^\[(targets|lift|render|protocol|vocab|void)\]|Error|Traceback' | tee -a $L; }
[ -e $T/basis.npz ] || run $V3/sam3_clip_targets.py --identity $N --out-dir $T ${KF:+--kf-dir $KF} ${FRUIT:+--fruit}
[ -e $A/features_lite.bin ] || run $V3/lift_pixel_features.py --identity $N --targets $T --out $A/features_lite.bin
[ -e $A/maps_lite/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $V3/splat_feature_render.py --config $CFG --features $A/features_lite.bin --frames $FR --out-dir $A/maps_lite --colmap-sparse $SPARSE
[ -s $N/protocol_lite.jsonl ] || run $V3/argmax_eval.py --identity $N --tag lite --space flat --basis $T/basis.npz --maps arms/maps_lite/%s.npz --void-ref $A/features_lite.bin $N/features_B.bin
say "LangSplat-lite done in $(( $(date +%s)-t0 )) s"
