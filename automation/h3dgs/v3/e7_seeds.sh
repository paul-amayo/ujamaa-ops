#!/bin/bash
# e7_seeds.sh — paper E7 (2026-10-13): the production embedder recipe under three seeds per citrus survey (object / row retrieval from the training
# log), then the downstream census seed re-built from each embedder on two cells (01 3_1, 05 1_0) and scored with the E1 protocol (tags seed1..3).
#   e7_seeds.sh   env: SEEDS="1 2 3"
set -uo pipefail; SEEDS=${SEEDS:-"1 2 3"}; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/e7_seeds.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); say "=== E7 seeds $SEEDS"
for SV in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal 05_13D_Jackal; do R=/home/paperspace/data/citrus_all/$SV; HJ=$R/prod/bateleur/scene_graph/marker_hierarchy.json; OUT=$R/experimental/identity/embedder; mkdir -p $OUT
  for s in $SEEDS; do EN=${SV}_v1g_s$s; [ -e $OUT/$EN/ckpts/model_best.pth ] && continue; t1=$(date +%s)
    (cd $NS && nice -n 5 pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name $EN --output-dir $OUT --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms --seed $s) > /home/paperspace/logs/e7_embedder_$EN.log 2>&1
    say "$EN rc=$? in $(( $(date +%s)-t1 )) s: $(tr '\r' '\n' < /home/paperspace/logs/e7_embedder_$EN.log | grep -a -E '^Object ' | tail -1 | tr -s ' ') | $(tr '\r' '\n' < /home/paperspace/logs/e7_embedder_$EN.log | grep -a 'Row (retrieval)' | tail -1 | tr -s ' ')"
  done; done
for N in /home/paperspace/data/citrus_all/01_13B_Jackal/experimental/v3/chunk_3_1/identity /home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity; do
  SV=$(echo $N | awk -F/ '{print $6}'); A=$N/arms; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); WS=$(dirname $N); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); [ -e $A/features_A.bin.labels.npy ] || { [ -e $N/features_B.bin.labels.npy ] || (cd $NS && pixi run python $V3/labels_from_seed.py --identity $N 2>&1 | grep -a labels); ln -sf $N/features_B.bin.labels.npy $A/features_A.bin.labels.npy; }
  for s in $SEEDS; do EMB=/home/paperspace/data/citrus_all/$SV/experimental/identity/embedder/${SV}_v1g_s$s/ckpts/model_best.pth; [ -e $EMB ] || continue
    [ -e $A/features_seed$s.bin ] || (cd $NS && pixi run python $V3/reseed_from_labels.py --labels $A/features_A.bin.labels.npy --embedder $EMB --out $A/features_seed$s.bin 2>&1 | grep -a reseed | tee -a $L)
    [ -e $A/maps_seed$s/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || (cd $NS && pixi run python $V3/splat_feature_render.py --config $CFG --features $A/features_seed$s.bin --frames $FR --out-dir $A/maps_seed$s --colmap-sparse $SPARSE 2>&1 | grep -a "render\] splatfacto")
    [ -s $N/protocol_seed$s.jsonl ] || (cd $NS && pixi run python $V3/argmax_eval.py --identity $N --tag seed$s --space hyper --emb $EMB --maps arms/maps_seed$s/%s.npz 2>&1 | grep -a "^\[protocol\]" | cut -c1-300 | tee -a $L)
  done; done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
