#!/bin/bash
# e3_tangent_chain.sh (2026-11-11, Paul: "why are we using the decoder for tree, that's different for HiGH and LERF"): HiGH "tree" through
# the encoder path, the letter's readout (--space hyper_tangent --category: cosines to g(word) and g(negatives), LERF relevancy), tag
# cat_tangent_vf, on the E3 cells. Log: /home/paperspace/logs/sec4/e3t.log
set -uo pipefail
B=/home/paperspace/logs/sec4/bin; LG=/home/paperspace/logs/sec4; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi
git -C /home/paperspace/code fetch -q origin && git -C /home/paperspace/code show origin/main:automation/h3dgs/v3/argmax_eval.py > $B/argmax_eval_e3t.py
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol)\]|Error|Traceback' | grep -av '^\[render\] kf' | tail -2 | tee -a $LG/e3t.log; }
for N in /home/paperspace/data/citrus_all/*/experimental/v3/*/identity; do
  [ -s $N/protocol_cat_lite.jsonl ] || continue; [ -s $N/protocol_cat_tangent_vf.jsonl ] && continue
  C=$(echo $N | sed "s#/home/paperspace/data/##; s#/#__#g"); M=/mnt/shared/voidfree_tmp/$C/e3t
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SP=$(dirname $N)/sparse/0; [ -e $SP/images.bin ] || SP=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  run $B/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $M --colmap-sparse $SP --void-free 0.5
  run $B/argmax_eval_e3t.py --identity $N --tag cat_tangent_vf --space hyper_tangent --category tree=ALL,orange=FRUIT --maps $M/%s.npz --out $N/protocol_cat_tangent_vf.jsonl; rm -rf $M
done
cd $LG && python3 $B/protocol_table.py --inventory /home/paperspace/logs/identity_inventory_v2.jsonl --tag cat_tangent_vf --min-frac 0.002 --compare cat_tangent_vf,cat_walk_vf,cat_lite --json $LG/e3_category_vf.json --md $LG/e3_category_vf.md 2>&1 | tail -1 | tee -a $LG/e3t.log
echo E3T DONE | tee -a $LG/e3t.log
