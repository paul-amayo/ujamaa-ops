#!/bin/bash
# re-seed the super field with the part share-assign rule (0.1 pre-registered = the production fruit constant; 0.5 as sensitivity), render, guard, score
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3
EMB=$W/experimental/identity/embedder/05_13D_super_v1/ckpts/model_best.pth; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed)|Error|Traceback' | cut -c1-600; }
echo "=== guard on the majority seed with waypoints at the level radii (max-dist 7.56)"; run $V3/argmax_eval.py --identity $N --tag super_md756 --space hyper --emb $EMB --maps $P/maps_super/%s.npz --max-dist 7.56 --out $P/protocol_super_md756.jsonl
for TAU in 0.1 0.5; do T=$(echo $TAU | tr -d '.'); echo "=== part share-assign $TAU"
  run $V3/build_super_seed.py --w-npz $P/W_super.npz --embedder $EMB --nodes-npz $P/nodes.npz --out-features $P/features_super_ps$T.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-share-assign $TAU
  mkdir -p $P/maps_super_ps$T; run $V3/splat_feature_render.py --config $CFG --features $P/features_super_ps$T.bin --frames $FR --out-dir $P/maps_super_ps$T --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback'
  for MD in 11.1 7.56; do M=$(echo $MD | tr -d '.'); echo "--- max-dist $MD"
    run $V3/argmax_eval.py --identity $N --tag super_ps${T}_md$M --space hyper --emb $EMB --maps $P/maps_super_ps$T/%s.npz --max-dist $MD --out $P/protocol_super_ps${T}_md$M.jsonl
    run $V3/super_eval.py --identity $N --parts $P --emb-super $EMB --maps-dir maps_super_ps$T --max-dist $MD --out $P/super_eval_ps${T}_md$M.json --dump $P/super_dump_ps${T}_md$M | grep -a 'heads n=\|Error\|Traceback'
  done
done
echo RESEED-DONE
