#!/bin/bash
# super hierarchy v2 (pre-registered 10-20): parts = trunk + branch only (no canopy node); parts seeded by the fruit recipe (floor 0.01, bg vs subtree, share-assign 0.1); embedder retrained; waypoints at 4 x the embedder's row radius
set -uo pipefail
S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad; while ! grep -aq 'PAF-DONE\|Traceback' $S/super_paf.log; do sleep 10; done
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; V=$P/v2; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH
EN=05_13D_super_v2; EMB=$W/experimental/identity/embedder/$EN/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); mkdir -p $V
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed|fig|nodes|hier|check|parts)|Row \(retrieval\)|Error|Traceback' | cut -c1-700; }
t0=$(date +%s); run $V3/parts_nodes.py $N $WS/images $P/train $V --types "tree trunk,branch"
python3 $V3/parts_hierarchy.py $W/prod/bateleur/scene_graph/marker_hierarchy.json $V/nodes.npz $V/marker_hierarchy_super.json
[ -e $EMB ] || run $HIGH/train_hyperembedder_graph.py --hierarchy-json $V/marker_hierarchy_super.json --experiment-name $EN --output-dir $W/experimental/identity/embedder --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms | tail -2
echo "[chain] embedder $(( $(date +%s)-t0 )) s"; run $V3/super_retrieval_check.py $EMB $N/text_bank.npz $V/nodes.npz $V/marker_hierarchy_super.json > $S/v2_check.log; cat $S/v2_check.log
MD=$(python3 -c "import re; t=open('$S/v2_check.log').read(); m=re.search(r'rows [\d.]+-[\d.]+ \(median ([\d.]+)\)', t); print(round(4*float(m.group(1)), 2))"); echo "[chain] waypoint radius max-dist $MD"
run $V3/splat_census.py --config $CFG --supervision-dir $V/supervision --names-json $N/split_names.json --with-bg --out-npz $V/W_super.npz | tail -1
run $V3/build_super_seed.py --w-npz $V/W_super.npz --embedder $EMB --nodes-npz $V/nodes.npz --out-features $V/features_super.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-floor 0.01 --bg-vs-subtree --part-share-assign 0.1
mkdir -p $V/maps_super; run $V3/splat_feature_render.py --config $CFG --features $V/features_super.bin --frames $FR --out-dir $V/maps_super --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback'
run $V3/argmax_eval.py --identity $N --tag super_v2 --space hyper --emb $EMB --maps $V/maps_super/%s.npz --max-dist $MD --out $V/protocol_super_v2.jsonl
mkdir -p $V/verdict; ln -sfn $P/verdict $V/verdict_link; for f in $P/verdict/*; do ln -sf $f $V/verdict/; done
run $V3/super_eval.py --identity $N --parts $V --emb-super $EMB --maps-dir maps_super --max-dist $MD --out $V/super_eval.json --dump $V/super_dump
python3 $V3/super_fig.py $V kf_002411.png 36 $V/nodes.npz $V/fig_super_kf002411_t36.png --dump-dir super_dump 2>&1 | grep -a '^\[fig\]'
echo "V2-DONE in $(( ($(date +%s)-t0)/60 )) min"
