#!/bin/bash
# super hierarchy on 05 1_0: embedder with part nodes -> census over the combined maps -> seed -> render the verdict frames
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3
HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; EN=05_13D_super_v1; EMB=$W/experimental/identity/embedder/$EN/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])")
S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad; mkdir -p $P/maps_super
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[|Row \(retrieval\)|Object \(retrieval\)|Error|Traceback|epoch (1|500|1000|1499)\b' | cut -c1-240; }
t0=$(date +%s)
if [ ! -e $EMB ]; then run $HIGH/train_hyperembedder_graph.py --hierarchy-json $P/marker_hierarchy_super.json --experiment-name $EN --output-dir $W/experimental/identity/embedder --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms > $S/super_embedder.log 2>&1; echo "[chain] embedder rc=$? in $(( $(date +%s)-t0 )) s"; grep -a 'retrieval' $S/super_embedder.log | tail -2 | cut -c1-200; fi
[ -e $EMB ] || { echo "[chain] embedder FAILED"; exit 1; }
[ -e $P/W_super.npz ] || run $V3/splat_census.py --config $CFG --supervision-dir $P/supervision --names-json $N/split_names.json --with-bg --out-npz $P/W_super.npz
run $V3/build_super_seed.py --w-npz $P/W_super.npz --embedder $EMB --nodes-npz $P/nodes.npz --out-features $P/features_super.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
run $V3/splat_feature_render.py --config $CFG --features $P/features_super.bin --frames $FR --out-dir $P/maps_super --colmap-sparse $WS/sparse/0
echo "[chain] done in $(( ($(date +%s)-t0)/60 )) min"
