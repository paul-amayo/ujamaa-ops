#!/bin/bash
# native_cell_test.sh — decisive test of the native H3DGS identity seed vs the containment side-car on 05 chunk 1_0
# (UJAMAA, 2026-10-03; plan in lab_notebook/2026-10.md). Census on the hierarchy's own nodes at the serving cut over the
# side-car's train frames -> census-init assignment with the side-car flags -> native feature maps; side-car feature maps
# at the same frames; both scored by the same containment_eval.py --features-npz code.
set -uo pipefail
S=/home/paperspace/data/citrus_all/05_13D_Jackal; P=$S/experimental/h3dgs_expo; CN=1_0
O=$S/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo; SUP=$O/supervision/trees_only
N=$S/experimental/h3dgs_native/chunk_1_0_expo
SCFG=$(ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/config.yml | head -1)
EMB=$S/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth; HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NT=/home/paperspace/code/automation/h3dgs/native; H3=/home/paperspace/code/hierarchical-3d-gaussians
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/native_census_05_1_0.log; VL=$N/verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export CUDA_HOME=/home/paperspace/code/_cuda12; export PATH=$CUDA_HOME/bin:$PATH
SCENE="-s $P/camera_calibration/chunks/$CN -m $P/output/trained_chunks/$CN --hierarchy $P/output/trained_chunks/$CN/hierarchy.hier_opt -i ../../rectified/images --eval --data_device cpu"
t0=$(date +%s); say "=== native census test, 05 chunk $CN (h3dgs_expo) vs side-car $(basename $(dirname $SCFG))"
python3 $NT/pick_frames.py --names $N/split_names.json --sup $SUP --must kf_002411.png kf_000025.png --n-train 8 --n-eval 4 --out $N/frames.json | tee -a $L
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $NT/sidecar_feature_dump.py --config $SCFG --frames $FR --out-dir $N/maps_sidecar --tag sidecar) > $L.dump 2>&1 &
DUMP=$!
t1=$(date +%s); (cd $H3 && $PYH $NT/hier_census.py $SCENE --supervision-dir $SUP --names-json $N/split_names.json --taus 3 --cut-width 1280 --with-bg --out-npz $N/W_tau3.npz) 2>&1 | grep -aE '^\[census\]|Error|Traceback' | tee -a $L
[ -e $N/W_tau3.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t1 )) s"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W_tau3.npz --embedder $EMB --out-features $N/features_census_tau3.bin --tree-floor 1.0 --bg-competes --bg-ratio 2) 2>&1 | grep -aE 'background|floors|assigned|node features|Error|Traceback' | tee -a $L
[ -e $N/features_census_tau3.bin ] || { say "assignment FAILED"; exit 1; }
(cd $H3 && $PYH $NT/hier_feature_render.py $SCENE --features $N/features_census_tau3.bin --frames $FR --taus 3 --out-dir $N/maps_native) 2>&1 | grep -aE '^\[render\]|Error|Traceback' | tee -a $L
wait $DUMP; grep -aE '^\[dump\]|Error|Traceback' $L.dump | tee -a $L
mkdir -p $N/figs; : > $VL
score(){ f=$1; m=$2; z=$3; (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/containment_eval.py --features-npz $z --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $f --kf-images $KF --out $N/figs/${f%.png}_$m.png 2>&1) | grep -aE '^(TREE|ROW|FRUIT) |norm gate' | sed "s/^/[$m $f] /"; }
export -f score; export N NS EMB ARU HJ SUP KF
for f in $FR; do echo "$f native $N/maps_native/${f%.png}_tau3.npz"; echo "$f sidecar $N/maps_sidecar/${f%.png}_sidecar.npz"; done | xargs -P 4 -L 1 bash -c 'score "$0" "$1" "$2"' >> $VL
say "scored $(grep -c '' $VL) lines; done in $(( ($(date +%s)-t0)/60 )) min"
