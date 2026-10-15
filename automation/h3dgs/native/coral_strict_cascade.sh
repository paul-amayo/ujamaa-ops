#!/bin/bash
# coral_strict_cascade.sh <survey> — swap a citrus survey's scene-graph rows to CORAL dir-0.985 (Paul 2026-10-13: "fix the diagonal rows";
# the strict hierarchy was built and checked first: experimental/coral_rows_20261013/strict_0985, 0 diagonal rows) and re-derive everything
# the paper measures from it. The old hierarchy and the v1g embedder are COPIED to experimental/era_census07_20261013/ and v1g stays in place
# (the lio_row100 block fleet still pairs with it: embedder-era pairing); the new survey embedder is <SV>_v1h. Per v3 cell: the old seed, bank,
# maps, arms and protocol files move to <identity>/era_census07_20261013/, then text bank -> seed B -> verdict renders -> E1 (tag v3). Then the
# Euclidean twin is retrained, arms B/C/C30/D run, and the seeding + category/lite arms on the survey's E2b cells. Resumable (existence checks).
set -uo pipefail; SV=$1; S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/bateleur; HJ=$B/scene_graph/marker_hierarchy.json; NEWH=$S/experimental/coral_rows_20261013/strict_0985/marker_hierarchy.json
ERA=$S/experimental/era_census07_20261013; V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native; ARU=/home/paperspace/code/aru_sil_core/src/scripts
HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; NS=/home/paperspace/code/nerf_new; EMB=$B/embedder/${SV}_v1h/ckpts/model_best.pth; INV=/home/paperspace/logs/identity_inventory_v2.jsonl
L=/home/paperspace/logs/coral_strict_cascade_$SV.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $SV: $*" | tee -a $L; }; t0=$(date +%s)
run(){ (cd $NS && nice -n 10 pixi run python "$@" 2>&1) | grep -aE '^\[(text-bank|render|protocol|vocab)\]|assigned|Error|Traceback' | tee -a $L; }
say "=== cascade start ($NEWH)"; mkdir -p $ERA
[ -e $ERA/marker_hierarchy.json ] || cp -a $HJ $ERA/marker_hierarchy.json; [ -d $ERA/embedder_${SV}_v1g ] || cp -a $B/embedder/${SV}_v1g $ERA/embedder_${SV}_v1g
python3 -c "import json; p=json.load(open('$HJ')).get('_provenance',{}); print(p.get('row_solver'), p.get('dominant_direction_threshold'))" | grep -q "coral 0.985" || { cp $NEWH $HJ; say "prod hierarchy <- strict 0.985 ($(python3 -c "import json; print(len(json.load(open('$HJ'))['rows']))") rows); old copy in $ERA"; }
if [ ! -e $EMB ]; then t1=$(date +%s); (cd $NS && nice -n 5 pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name ${SV}_v1h --output-dir $B/embedder --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms) > /home/paperspace/logs/embedder_${SV}_v1h.log 2>&1
  say "embedder ${SV}_v1h rc=$? in $(( $(date +%s)-t1 )) s: $(tr '\r' '\n' < /home/paperspace/logs/embedder_${SV}_v1h.log | grep -a -E '^Object ' | tail -1 | tr -s ' ') | $(tr '\r' '\n' < /home/paperspace/logs/embedder_${SV}_v1h.log | grep -a 'Row (retrieval)' | tail -1 | tr -s ' ')"; fi
[ -e $EMB ] || { say "embedder FAILED"; exit 1; }
grep "\"survey\": \"$SV\"" $INV > /home/paperspace/logs/identity_inventory_$SV.jsonl
while read -r line; do eval $(python3 -c "import json,sys; r=json.loads(sys.argv[1]); print(f\"N={r['identity']} SUP={r['sup']} WS={r['ws']}\")" "$line"); E=$N/era_census07_20261013; mkdir -p $E
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0; FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  if ! grep -q "_v1h" $N/features_B.bin.json 2>/dev/null; then for f in features_B.bin features_B.bin.json features_B.bin.labels.npy text_bank.npz maps_v3 arms maps_fruit maps_fruit_full verdicts.log; do [ -e $N/$f ] && mv $N/$f $E/; done; for f in $N/protocol_*.jsonl; do [ -e "$f" ] && mv "$f" $E/; done; fi
  [ -e $N/text_bank.npz ] || run $NT/build_text_bank.py --manifest $SUP/manifest.json --hierarchy-json $HJ --out $N/text_bank.npz
  [ -e $N/features_B.bin ] || (cd $NS && HIGH_EMBEDDER_CKPT=$EMB nice -n 10 pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_B.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1 2>&1) | grep -aE 'assigned|Error|Traceback' | tee -a $L
  [ -e $N/maps_v3/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $V3/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $N/maps_v3 --colmap-sparse $SPARSE
  [ -s $N/protocol_v3.jsonl ] || run $V3/argmax_eval.py --identity $N --tag v3 --space hyper
  say "cell $(basename $WS) re-seeded + scored"
done < /home/paperspace/logs/identity_inventory_$SV.jsonl
T=$S/experimental/identity/embedder/${SV}_euclid_v1; [ -d $T ] && [ ! -d $ERA/embedder_${SV}_euclid_v1 ] && mv $T $ERA/embedder_${SV}_euclid_v1
bash $V3/e2_embedders.sh /home/paperspace/logs/identity_inventory_$SV.jsonl > /dev/null 2>&1; say "euclidean twin: $(ls $T/ckpts/model_best.pth 2>/dev/null || echo MISSING)"
ARMS=B,C,C30,D bash $V3/e2_arms_run.sh /home/paperspace/logs/identity_inventory_$SV.jsonl > /dev/null 2>&1; say "arms B,C,C30,D done"
grep "/$SV/" /home/paperspace/logs/e2b_cells.txt > /home/paperspace/logs/e2b_cells_$SV.txt; bash $V3/e2b_seed_run.sh /home/paperspace/logs/e2b_cells_$SV.txt > /dev/null 2>&1; bash $V3/e3_lite_run.sh /home/paperspace/logs/e2b_cells_$SV.txt > /dev/null 2>&1
say "=== cascade done in $(( ($(date +%s)-t0)/60 )) min"
