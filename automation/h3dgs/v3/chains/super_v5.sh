#!/bin/bash
# v5 (pre-registered 10-20): v4 with the part-view census visibility-weighted (a view's part mass counts through the gaussian's transmittance), so the part share belongs to the primitives seen AT the part; success = 3D localisation (trunk-node centroids within 0.5 m of the markers, tight spread) beside the pixel IoU
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; V4=$P/v4; V=$P/v5; NS=/home/paperspace/code/nerf_new; T=/home/paperspace/code/automation/h3dgs/v3
EMB=$W/experimental/identity/embedder/05_13D_super_v4/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad
MD=$(python3 -c "import re; t=open('$S/v4_check.log').read(); m=re.search(r'rows [\d.]+-[\d.]+ \(median ([\d.]+)\)', t); print(round(4*float(m.group(1)), 2))"); mkdir -p $V/verdict $V/maps_super; ln -sf $V4/nodes.npz $V/nodes.npz; for f in $P/verdict/*; do ln -sf $f $V/verdict/; done
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed|fig|census|loc)|Error|Traceback' | cut -c1-700; }
t0=$(date +%s); run $T/splat_census.py --config $CFG --supervision-dir $V4/supervision_partview --names-json $N/split_names.json --with-bg --vis-weight --out-npz $V/W_part_vis.npz | tail -2
run $T/build_super_seed.py --w-npz $V4/W_super.npz --w-part-npz $V/W_part_vis.npz --embedder $EMB --nodes-npz $V4/nodes.npz --out-features $V/features_super.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-floor 0.01 --bg-vs-subtree --part-share-assign 0.1 --promote-withheld
run $T/super_localise.py $N $V/features_super.bin.labels.npy $V4/nodes.npz $V/trunk_localisation.json
run $T/splat_feature_render.py --config $CFG --features $V/features_super.bin --frames $FR --out-dir $V/maps_super --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback'
run $T/argmax_eval.py --identity $N --tag super_v5 --space hyper --emb $EMB --maps $V/maps_super/%s.npz --max-dist $MD --out $V/protocol_super_v5.jsonl
run $T/super_eval.py --identity $N --parts $V --emb-super $EMB --maps-dir maps_super --max-dist $MD --out $V/super_eval.json --dump $V/super_dump
run $T/super_proj_diag.py $N $V x 0 $V/features_super.bin.labels.npy --all-heads $V/super_eval.json --out $V/proj_heads.json | tail -3
python3 $T/super_fig.py $V kf_003289.png 0 $V4/nodes.npz $V/fig_super_kf003289_t0.png --dump-dir super_dump 2>&1 | grep -a '^\[fig\]'
echo "V5-DONE in $(( ($(date +%s)-t0)/60 )) min"
