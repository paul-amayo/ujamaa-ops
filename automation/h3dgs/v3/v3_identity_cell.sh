#!/bin/bash
# v3_identity_cell.sh — recipe v3 identity on one chunk (2026-10-08): flat-gaussian census on the splatfacto checkpoint, the native
# seed-B assignment (build_census_init.py, unchanged), feature maps for the native cell's verdict frames, containment verdicts by the
# same scorer (score_maps.sh / containment_eval.py --features-npz) against the native H3DGS cell's rows (same supervision, frames,
# embedder) -> table. The parity check of plans/recipe_v3_splatfacto.md item 3.
# Env: CFG (splatfacto run config.yml), CELL (native cell: supervision/trees_only, split_names.json, frames.json, verdicts.log),
#      EMB, HJ, KF, OUT; [FLAGS] seed flags (default seed B), [NATIVE_TAG=B_bg2share], [SPARSE] chunk sparse/0 for off-workspace frames
set -uo pipefail
: ${CFG:?} ${CELL:?} ${EMB:?} ${HJ:?} ${KF:?} ${OUT:?}
FLAGS=${FLAGS:---tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1}; NATIVE_TAG=${NATIVE_TAG:-B_bg2share}
V3=/home/paperspace/code/automation/h3dgs/v3; NT=/home/paperspace/code/automation/h3dgs/native; ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
SUP=$CELL/supervision/trees_only; N=$OUT; mkdir -p $N; SPARSE=${SPARSE:-$(dirname $(dirname $(dirname $(dirname $CFG))))/../sparse/0}   # <ws>/outputs/<run>/splatfacto/run/config.yml -> <ws>/sparse/0
L=/home/paperspace/logs/v3_identity_$(echo $OUT | cut -d/ -f6)_$(basename $(dirname $OUT)).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
cp -n $CELL/frames.json $CELL/split_names.json $N/ 2>/dev/null; [ -e $CELL/text_bank.npz ] && cp -n $CELL/text_bank.npz $N/
t0=$(date +%s); say "=== v3 identity: $N | ckpt $CFG | sup $SUP | emb $EMB | native rows $NATIVE_TAG from $CELL"
(cd $NS && pixi run python $V3/splat_census.py --config $CFG --supervision-dir $SUP --names-json $N/split_names.json --with-bg --out-npz $N/W.npz) 2>&1 | grep -aE '^\[census\] (saved|splat|sf)|Error|Traceback' | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t0 )) s"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_B.bin $FLAGS) 2>&1 | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
[ -e $N/features_B.bin ] || { say "assignment FAILED"; exit 1; }
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
(cd $NS && pixi run python $V3/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $N/maps_v3 --colmap-sparse $SPARSE) 2>&1 | grep -aE '^\[render\]|Error|Traceback' | sed -E 's/ -> .*//' | tee -a $L
grep -a "^\[$NATIVE_TAG " $CELL/verdicts.log > $N/verdicts.log 2>/dev/null || : > $N/verdicts.log
bash $NT/score_maps.sh $N v3 maps_v3/%s.npz $SUP $EMB $HJ $KF 4 | tee -a $L
if grep -aq "^\[$NATIVE_TAG " $N/verdicts.log; then   # a native H3DGS cell scored the same frames: side-by-side table
  python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json --models v3 $NATIVE_TAG > $N/table_v3_vs_native.md; tail -8 $N/table_v3_vs_native.md | tee -a $L
else   # fleet chunks without a native cell: v3 means by kind and split
  python3 - $N <<'PY' | tee -a $L
import json, re, sys; N = sys.argv[1]; ev = set(json.load(open(f'{N}/split_names.json'))['eval']); acc = {}
for line in open(f'{N}/verdicts.log', errors='ignore'):
    m = re.match(r'\[v3 (\S+)\] (TREE|ROW|FRUIT)\b.*IoU ([\d.]+)', line)
    if m: acc.setdefault((m.group(2), 'eval' if m.group(1) in ev else 'train'), []).append(float(m.group(3)))
print('[v3 means] ' + '; '.join(f'{k[0]} {k[1]}: {len(v)} heads, mean IoU {sum(v) / len(v):.3f}' for k, v in sorted(acc.items())))
PY
fi
say "done in $(( ($(date +%s)-t0)/60 )) min"
