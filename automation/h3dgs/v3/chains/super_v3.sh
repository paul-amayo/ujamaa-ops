#!/bin/bash
# v3 (pre-registered 10-20, queued before v2's result): v2's nodes and embedder; the part share judged only in the views that observed the part (part-view census)
set -uo pipefail
S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad; while ! grep -aq 'V2-DONE\|Traceback' $S/super_v2.log; do sleep 10; done
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; V2=$P/v2; V3=$P/v3; NS=/home/paperspace/code/nerf_new; T=/home/paperspace/code/automation/h3dgs/v3
EMB=$W/experimental/identity/embedder/05_13D_super_v2/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); mkdir -p $V3/verdict $V3/maps_super
MD=$(python3 -c "import re; t=open('$S/v2_check.log').read(); m=re.search(r'rows [\d.]+-[\d.]+ \(median ([\d.]+)\)', t); print(round(4*float(m.group(1)), 2))")
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed|fig|nodes|census)|Error|Traceback' | cut -c1-700; }
t0=$(date +%s); run $T/parts_nodes.py $N $WS/images $P/train $V3 --types "tree trunk,branch" --partview | grep -a nodes
ln -sf $V2/nodes.npz $V3/nodes.npz; for f in $P/verdict/*; do ln -sf $f $V3/verdict/; done
run $T/splat_census.py --config $CFG --supervision-dir $V3/supervision_partview --names-json $N/split_names.json --with-bg --out-npz $V3/W_part.npz | tail -1
run $T/build_super_seed.py --w-npz $V2/W_super.npz --w-part-npz $V3/W_part.npz --embedder $EMB --nodes-npz $V2/nodes.npz --out-features $V3/features_super.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-floor 0.01 --bg-vs-subtree --part-share-assign 0.1
run $T/splat_feature_render.py --config $CFG --features $V3/features_super.bin --frames $FR --out-dir $V3/maps_super --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback'
run $T/argmax_eval.py --identity $N --tag super_v3 --space hyper --emb $EMB --maps $V3/maps_super/%s.npz --max-dist $MD --out $V3/protocol_super_v3.jsonl
run $T/super_eval.py --identity $N --parts $V3 --emb-super $EMB --maps-dir maps_super --max-dist $MD --out $V3/super_eval.json --dump $V3/super_dump
python3 $T/super_fig.py $V3 kf_002411.png 36 $V2/nodes.npz $V3/fig_super_kf002411_t36.png --dump-dir super_dump 2>&1 | grep -a '^\[fig\]'
echo "V3-DONE in $(( ($(date +%s)-t0)/60 )) min"
