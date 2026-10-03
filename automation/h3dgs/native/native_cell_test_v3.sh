#!/bin/bash
# native_cell_test_v3.sh (v2 + PICKOPT for pick_frames, e.g. --prefer-fruit) — native H3DGS identity seed vs the containment side-car on one cell (UJAMAA, 2026-10-03).
# v2 of native_cell_test.sh: parametrised by env, and CUDA_HOME=_cuda12 is scoped to the h3dgs-env commands ONLY
# (v1 exported it chain-wide; the nerf_new side-car dump inherited it and rebuilt nerf_new's gsplat JIT cache against
# CUDA 12 -> libcudart.so.12 ImportError; see lab_notebook 2026-10-03).
# Env: SV (survey root), PROJ (H3DGS project), CN (chunk), SC (side-car dir), SUPN (supervision subdir, trees_only),
#      EMB, HJ, KF, OUT (native cell dir), MUST (verdict frames to include), NTR (8), NEV (4), TAUS ("3"), FLAGS (assignment)
set -uo pipefail
: ${SV:?} ${PROJ:?} ${CN:?} ${SC:?} ${EMB:?} ${HJ:?} ${KF:?} ${OUT:?}
SUPN=${SUPN:-trees_only}; PICKOPT=${PICKOPT:-}; MUST=${MUST:-}; NTR=${NTR:-8}; NEV=${NEV:-4}; TAUS=${TAUS:-3}
FLAGS=${FLAGS:---tree-floor 1.0 --bg-competes --bg-ratio 2}; IMGS=${IMGS:-../../rectified/images}
SUP=$SC/supervision/$SUPN; N=$OUT; mkdir -p $N
SCFG=$(ls $SC/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2*/high/*/config.yml 2>/dev/null | head -1); SCFG=${SIDECAR_CFG:-$SCFG}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NT=/home/paperspace/code/automation/h3dgs/native; H3=/home/paperspace/code/hierarchical-3d-gaussians
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new
HENV="env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH"
L=/home/paperspace/logs/native_census_$(basename $(dirname $(dirname $N)))_$(basename $N).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
CUTW=${CUTW:-$(python3 -c "
import struct,sys; f=open(sys.argv[1],'rb'); n=struct.unpack('<Q',f.read(8))[0]; cid,model,w,h=struct.unpack('<iiQQ',f.read(24)); print(w)" $PROJ/camera_calibration/chunks/$CN/sparse/0/cameras.bin)}
SCENE="-s $PROJ/camera_calibration/chunks/$CN -m $PROJ/output/trained_chunks/$CN --hierarchy $PROJ/output/trained_chunks/$CN/hierarchy.hier_opt -i $IMGS --eval --data_device cpu"
t0=$(date +%s); say "=== native census v2: $N | hierarchy $PROJ chunk $CN (cut width $CUTW) | side-car $SCFG | sup $SUP | emb $EMB"
[ -e $N/split_names.json ] || (cd $NS && pixi run python $NT/sidecar_split_names.py --config $SCFG --out $N/split_names.json) 2>&1 | grep -a '^\[split\]' | tee -a $L
python3 $NT/pick_frames.py --names $N/split_names.json --sup $SUP --must $MUST --n-train $NTR --n-eval $NEV $PICKOPT --out $N/frames.json | tee -a $L
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $NT/sidecar_feature_dump.py --config $SCFG --frames $FR --out-dir $N/maps_sidecar --tag sidecar) > $L.dump 2>&1 &
DUMP=$!
t1=$(date +%s); (cd $H3 && $HENV $PYH $NT/hier_census.py $SCENE --supervision-dir $SUP --names-json $N/split_names.json --taus $TAUS --cut-width $CUTW --with-bg --out-npz $N/W.npz) 2>&1 | grep -aE '^\[census\] (saved|hierarchy)|Error|Traceback' | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t1 )) s"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_census.bin $FLAGS) 2>&1 | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
[ -e $N/features_census.bin ] || { say "assignment FAILED"; exit 1; }
(cd $H3 && $HENV $PYH $NT/hier_feature_render.py $SCENE --features $N/features_census.bin --frames $FR --taus 3 --cut-width $CUTW --out-dir $N/maps_native) 2>&1 | grep -aE '^\[render\]|Error|Traceback' | sed -E 's/ -> .*//' | tee -a $L
bash $NT/score_maps.sh $N native maps_native/%s_tau3.npz $SUP $EMB $HJ $KF 4 | tee -a $L
wait $DUMP; grep -aE '^\[dump\]|Error|Traceback' $L.dump | sed -E 's/ -> .*//' | tee -a $L
bash $NT/score_maps.sh $N sidecar maps_sidecar/%s_sidecar.npz $SUP $EMB $HJ $KF 4 | tee -a $L
python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json > $N/table.md; tail -8 $N/table.md | tee -a $L
say "done in $(( ($(date +%s)-t0)/60 )) min"
