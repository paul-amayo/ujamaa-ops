#!/bin/bash
# LoD route for the super hierarchy (Paul 10-20: "I'll buy the conceptual improvement in the trunks through LOD even if it's a lower IoU bar, it's still localisable, right"):
# the v8 guided-split trainer (fruit_lod_cell_v8.sh recipe: fruit protection, direct split, guided split until 25k, L1 weight 1+10, 30k, bilateral grid) on 05 1_0 with
# trunk ∪ fruit as the small-object class read directly (HIGH_FRUIT_MASK_DIR = parts/v4/fruitmask_trunk: 2 = trunk-node or fruit px, 1 = other labelled, 0 = unlabelled),
# then the v5 super pipeline on the NEW geometry: census over the combined maps + visibility-weighted part-view census, v4 seed rules, embedder super_v4 (waypoints at its
# radii), 3D localisation vs the markers, identity guard, composed "trunk of tree k", projection diagnostic, figures. Pre-registered expectation: composed median IoU 0.2–0.35,
# trunk-node centroids within 0.5 m of the markers for most heads, spread well under the v4/v5 2.2–2.3 m.  One checkpoint (disk at 17 GB free).
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; SF=$W/experimental/splatfacto_1_0; CELL=$W/experimental/h3dgs_native/chunk_1_0_sam3; N=$WS/identity; P=$WS/parts; V4=$P/v4; V=$P/lod
NS=/home/paperspace/code/nerf_new; T=/home/paperspace/code/automation/h3dgs/v3; S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad
EMB=$W/experimental/identity/embedder/05_13D_super_v4/ckpts/model_best.pth; EMBP=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])")
IT=${IT:-30000}; EXP=${EXP:-high_lodtrunk_${IT}_bilateral}; CFG=$SF/outputs/$EXP/high/run/config.yml; L=/home/paperspace/logs/lod_trunk_05_1_0.log
DPS=$(python3 -c "import json; print(json.load(open('$SF/outputs/sf_60000_bilateral/splatfacto/run/dataparser_transforms.json'))['scale'])"); m(){ python3 -c "print($1*$DPS)"; }
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); MD=$(python3 -c "import re; t=open('$S/v4_check.log').read(); m=re.search(r'rows [\d.]+-[\d.]+ \(median ([\d.]+)\)', t); print(round(4*float(m.group(1)), 2))" 2>/dev/null || echo 7.84)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; run(){ (cd $NS && pixi run python "$@") 2>&1 | grep -aE '^\[(super|protocol|render|super-seed|fig|census|loc|proj)|Error|Traceback' | cut -c1-700; }
mkdir -p $V/verdict $V/maps_super; ln -sf $V4/nodes.npz $V/nodes.npz; for f in $P/verdict/*; do ln -sf $f $V/verdict/; done; t0=$(date +%s)
say "=== trunk LoD: $EXP | dataparser scale $DPS | masks $V4/fruitmask_trunk | sup $CELL/supervision/trees_only | trainer embedder $(basename $(dirname $(dirname $EMBP))) | super embedder super_v4, max-dist $MD | df $(df -h / | tail -1 | awk '{print $4}') free"
if [ ! -e $CFG ]; then t1=$(date +%s)
  (cd $NS && env HIGH_EMBEDDER_CKPT=$EMBP HIGH_FRUIT_MASK_DIR=$V4/fruitmask_trunk HIGH_LOD_LOSS_W=10 HIGH_SEED_DUMP=$SF/outputs/$EXP/fruit_seeds.npz PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 pixi run ns-train high --data $SF --output-dir $SF/outputs --experiment-name $EXP --timestamp run --vis tensorboard \
     --max-num-iterations $IT --steps-per-save $IT --save-only-latest-checkpoint True --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 --pipeline.model.use-bilateral-grid True \
     --pipeline.model.enable-high-features False --pipeline.model.fruit-protect True --pipeline.model.fruit-tally-every 4 --pipeline.model.fruit-protect-tau 3 --pipeline.model.fruit-protect-min-frames 3 --pipeline.model.fruit-protect-share 0.01 --pipeline.model.fruit-anchor-weight 0.0 \
     --pipeline.model.fruit-densify True --pipeline.model.fruit-densify-direct-split True --pipeline.model.fruit-densify-tail 0 --pipeline.model.stop-split-at 15000 --pipeline.model.fruit-densify-scale $(m 0.02) --pipeline.model.fruit-densify-max-scale $(m 0.5) \
     --pipeline.model.fruit-guided-split True --pipeline.model.fruit-guided-min-size $(m 0.01) --pipeline.model.fruit-guided-until 25000 --pipeline.model.fruit-guided-lo 0.05 --pipeline.model.fruit-guided-hi 0.9 --pipeline.datamanager.semantic-dir $CELL/supervision/trees_only \
     nerfstudio-data --eval-mode filename) > $L.train 2>&1
  say "train rc=$? in $(( ($(date +%s)-t1)/60 )) min: $(grep -a -c '\[fruit-densify\] step' $L.train) split lines, guided $(grep -a -c '\[guided-split\]' $L.train), lod-loss $(grep -a -c '\[lod-loss\]' $L.train); masks: $(grep -a 'fruit masks:' $L.train | head -1 | cut -c1-120); last guided: $(grep -a '\[guided-split\]' $L.train | tail -1 | cut -c1-200)"
fi
[ -e $CFG ] || { say "no config — training FAILED (see $L.train)"; exit 1; }
run $T/splat_census.py --config $CFG --supervision-dir $V4/supervision --names-json $N/split_names.json --with-bg --out-npz $V/W_super.npz | tail -1 | tee -a $L
run $T/splat_census.py --config $CFG --supervision-dir $V4/supervision_partview --names-json $N/split_names.json --with-bg --vis-weight --out-npz $V/W_part_vis.npz | tail -1 | tee -a $L
run $T/build_super_seed.py --w-npz $V/W_super.npz --w-part-npz $V/W_part_vis.npz --embedder $EMB --nodes-npz $V4/nodes.npz --out-features $V/features_super.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-floor 0.01 --fruit-share-assign 0.1 --part-floor 0.01 --bg-vs-subtree --part-share-assign 0.1 --promote-withheld | tee -a $L
run $T/super_localise.py $N $V/features_super.bin.labels.npy $V4/nodes.npz $V/trunk_localisation.json --config $CFG | tee -a $L
run $T/splat_feature_render.py --config $CFG --features $V/features_super.bin --frames $FR --out-dir $V/maps_super --colmap-sparse $WS/sparse/0 | grep -a 'splatfacto\|Error\|Traceback' | tee -a $L
run $T/argmax_eval.py --identity $N --tag super_lod --space hyper --emb $EMB --maps $V/maps_super/%s.npz --max-dist $MD --out $V/protocol_super_lod.jsonl | tee -a $L
run $T/super_eval.py --identity $N --parts $V --emb-super $EMB --maps-dir maps_super --max-dist $MD --out $V/super_eval.json --dump $V/super_dump | tee -a $L
run $T/super_proj_diag.py $N $V x 0 $V/features_super.bin.labels.npy --all-heads $V/super_eval.json --out $V/proj_heads.json --config $CFG | tail -3 | tee -a $L
for h in "kf_003289.png 0" "kf_002411.png 36"; do set -- $h; python3 $T/super_fig.py $V $1 $2 $V4/nodes.npz $V/fig_super_${1%.png}_t$2.png --dump-dir super_dump 2>&1 | grep -a '^\[fig\]' | tee -a $L; done
say "LOD-DONE in $(( ($(date +%s)-t0)/60 )) min"
