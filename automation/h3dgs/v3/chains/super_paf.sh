#!/bin/bash
# parts-as-fruit seeding rule (pre-registered 10-20): part floor 0.01, background vs the subtree mass; then render, guard, score, figure
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3
EMB=$W/experimental/identity/embedder/05_13D_super_v1/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed|fig)|Error|Traceback' | cut -c1-700; }
T=paf; run $V3/build_super_seed.py --w-npz $P/W_super.npz --embedder $EMB --nodes-npz $P/nodes.npz --out-features $P/features_super_$T.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-floor 0.01 --bg-vs-subtree
mkdir -p $P/maps_super_$T; run $V3/splat_feature_render.py --config $CFG --features $P/features_super_$T.bin --frames $FR --out-dir $P/maps_super_$T --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback'
for MD in 7.56 11.1; do M=$(echo $MD | tr -d '.'); echo "--- max-dist $MD"
  run $V3/argmax_eval.py --identity $N --tag super_${T}_md$M --space hyper --emb $EMB --maps $P/maps_super_$T/%s.npz --max-dist $MD --out $P/protocol_super_${T}_md$M.jsonl
  run $V3/super_eval.py --identity $N --parts $P --emb-super $EMB --maps-dir maps_super_$T --max-dist $MD --out $P/super_eval_${T}_md$M.json --dump $P/super_dump_${T}_md$M
done
python3 $V3/super_fig.py $P kf_002411.png 36 $P/nodes.npz $P/fig_super_kf002411_t36.png --dump-dir super_dump_${T}_md756 2>&1 | grep -a '^\[fig\]'
echo PAF-DONE
