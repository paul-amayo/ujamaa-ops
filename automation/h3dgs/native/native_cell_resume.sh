#!/bin/bash
# native_cell_resume.sh — finish a native_cell_test_v2 cell whose census failed (UJAMAA, 2026-10-03): census ->
# assignment -> native render -> native scores, then wait for an already-running side-car dump (DUMP_PID) and score it.
# Same env as native_cell_test_v2.sh (SV PROJ CN SC SUPN EMB HJ KF OUT FLAGS TAUS CUTW IMGS) + DUMP_PID.
set -uo pipefail
: ${PROJ:?} ${CN:?} ${SC:?} ${EMB:?} ${HJ:?} ${KF:?} ${OUT:?}
SUPN=${SUPN:-trees_only}; TAUS=${TAUS:-3}; FLAGS=${FLAGS:---tree-floor 1.0 --bg-competes --bg-ratio 2}; IMGS=${IMGS:-../../rectified/images}; CUTW=${CUTW:-1280}
SUP=$SC/supervision/$SUPN; N=$OUT; ARU=/home/paperspace/code/aru_sil_core/src/scripts; NT=/home/paperspace/code/automation/h3dgs/native
H3=/home/paperspace/code/hierarchical-3d-gaussians; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
HENV="env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH"
L=/home/paperspace/logs/native_census_resume_$(basename $N).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
SCENE="-s $PROJ/camera_calibration/chunks/$CN -m $PROJ/output/trained_chunks/$CN --hierarchy $PROJ/output/trained_chunks/$CN/hierarchy.hier_opt -i $IMGS --eval --data_device cpu"
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
t0=$(date +%s); say "=== resume $N (census onward; side-car dump pid ${DUMP_PID:-none})"
(cd $H3 && $HENV $PYH $NT/hier_census.py $SCENE --supervision-dir $SUP --names-json $N/split_names.json --taus $TAUS --cut-width $CUTW --with-bg --out-npz $N/W.npz) > $L.census 2>&1
grep -aE '^\[census\] (saved|hierarchy)|Error|Traceback' $L.census | cut -c1-300 | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED (see $L.census)"; exit 1; }
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_census.bin $FLAGS) 2>&1 | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
[ -e $N/features_census.bin ] || { say "assignment FAILED"; exit 1; }
(cd $H3 && $HENV $PYH $NT/hier_feature_render.py $SCENE --features $N/features_census.bin --frames $FR --taus 3 --cut-width $CUTW --out-dir $N/maps_native) 2>&1 | grep -aE '^\[render\]|Error|Traceback' | sed -E 's/ -> .*//' | tee -a $L
bash $NT/score_maps.sh $N native maps_native/%s_tau3.npz $SUP $EMB $HJ $KF 4 | tee -a $L
[ -n "${DUMP_PID:-}" ] && while kill -0 $DUMP_PID 2>/dev/null; do sleep 15; done
say "side-car maps: $(ls $N/maps_sidecar 2>/dev/null | wc -l) of $(echo $FR | wc -w)"
bash $NT/score_maps.sh $N sidecar maps_sidecar/%s_sidecar.npz $SUP $EMB $HJ $KF 4 | tee -a $L
python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json > $N/table.md; sed -n '/| kind/,/^$/p' $N/table.md | tee -a $L
say "done in $(( ($(date +%s)-t0)/60 )) min"
