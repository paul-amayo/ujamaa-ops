#!/bin/bash
# one-frame decodes of every arm of 05 1_0 for the paper's qualitative strips (argmax_eval.py --dump-pred / --dump-cat)
set -uo pipefail
N=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity; A=$N/arms; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad; FJ=$S/one_frame.json
CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); WS=$(dirname $N); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
EMB_EUCLID=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/identity/embedder/05_13D_Jackal_euclid_v1/ckpts/model_best.pth
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(render|protocol|dump)|Error|Traceback' | cut -c1-220; }
for arm in zero3000 refine3000; do [ -e $A/maps_$arm/kf_002411.npz ] || run $V3/splat_feature_render.py --config $CFG --features $A/features_$arm.bin --frames kf_002411.png --out-dir $A/maps_$arm --colmap-sparse $SPARSE; done
run $V3/argmax_eval.py --identity $N --tag dump_v3 --space hyper --frames-json $FJ --out $S/dump_v3.jsonl --dump-pred $N/dumps/v3
run $V3/argmax_eval.py --identity $N --tag dump_hyper_own --space hyper_own --frames-json $FJ --out $S/dump_hyper_own.jsonl --dump-pred $N/dumps/hyper_own
run $V3/argmax_eval.py --identity $N --tag dump_euclid --space euclid --emb $EMB_EUCLID --maps arms/maps_euclid/%s.npz --frames-json $FJ --out $S/dump_euclid.jsonl --dump-pred $N/dumps/euclid
run $V3/argmax_eval.py --identity $N --tag dump_flat96 --space flat --basis $A/features_flat96.bin.basis.npz --maps arms/maps_flat96/%s.npz --frames-json $FJ --out $S/dump_flat96.jsonl --dump-pred $N/dumps/flat96
for arm in zero3000 refine3000 projvote; do run $V3/argmax_eval.py --identity $N --tag dump_$arm --space hyper --maps arms/maps_$arm/%s.npz --frames-json $FJ --out $S/dump_$arm.jsonl --dump-pred $N/dumps/$arm; done
run $V3/argmax_eval.py --identity $N --tag dump_cat_v3 --space hyper --category tree=ALL,orange=FRUIT --frames-json $FJ --out $S/dump_cat_v3.jsonl --dump-cat $N/dumps/cat_v3
run $V3/argmax_eval.py --identity $N --tag dump_cat_flat96 --space flat --basis $A/features_flat96.bin.basis.npz --maps arms/maps_flat96/%s.npz --category tree=ALL,orange=FRUIT --frames-json $FJ --out $S/dump_cat_flat96.jsonl --dump-cat $N/dumps/cat_flat96
echo DUMPS-DONE
