#!/bin/bash
# e4_fruit_cell.sh — the fruit level measured where fruit is (paper E4, 2026-10-13): pick the cell's fruit-richest frames (pick_frames.py --prefer-fruit,
# 4 census / 10 held-out), render the existing v3 seed on them and score with the E1 protocol (tag v3_fruit). Instance recall (fraction of fruit heads
# with any lit pixel) is read from the per-head records by e4_fruit_table.py.   N=<ws>/identity [NTRAIN=4] [NEVAL=10] e4_fruit_cell.sh
set -uo pipefail; : ${N:?}; V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native; NS=/home/paperspace/code/nerf_new
CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); WS=$(dirname $N); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
SUP=$(python3 -c "import numpy as np, json, os; print(os.path.dirname(json.loads(str(np.load('$N/text_bank.npz', allow_pickle=True)['meta']))['manifest']))")
L=/home/paperspace/logs/e4_fruit.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(echo $N | awk -F/ '{print $6"/"$8}'): $*" | tee -a $L; }; t0=$(date +%s)
run(){ (cd $NS && nice -n 10 pixi run python "$@" 2>&1) | grep -aE '^\[(render|protocol|vocab|void)\]|Error|Traceback' | tee -a $L; }
[ -e $N/frames_fruit.json ] || python3 $NT/pick_frames.py --names $N/split_names.json --sup $SUP --prefer-fruit --n-train ${NTRAIN:-4} --n-eval ${NEVAL:-10} --out $N/frames_fruit.json | tee -a $L
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames_fruit.json'))['frames']))"); NF=$(echo $FR | wc -w)
NEW=$(for f in $FR; do [ -e $N/maps_fruit/${f%.png}.npz ] || [ -e $N/maps_v3/${f%.png}.npz ] || echo $f; done)
[ -z "$NEW" ] || run $V3/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $NEW --out-dir $N/maps_fruit --colmap-sparse $SPARSE
for f in $FR; do [ -e $N/maps_fruit/${f%.png}.npz ] || ln -sf $N/maps_v3/${f%.png}.npz $N/maps_fruit/${f%.png}.npz; done   # frames already rendered for E1 are reused
[ -s $N/protocol_v3_fruit.jsonl ] || run $V3/argmax_eval.py --identity $N --tag v3_fruit --space hyper --maps maps_fruit/%s.npz --frames-json $N/frames_fruit.json
say "fruit frames $NF scored in $(( $(date +%s)-t0 )) s"
