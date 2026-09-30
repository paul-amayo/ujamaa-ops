#!/bin/bash
# sidecar_fruit_cuts.sh <survey> <side-car dir name> <chunk> [max fruits=6] — per-fruit containment verdicts on a densified
# chunk side-car, then the fitted thresholds as JSON for the app (Paul, 2026-09-30: "fruit is by containment"; the app lights
# a fruit at its own verdict threshold instead of the render service's per-frame Otsu, which lit whole canopies).
# The densify step scores ONE frame (the top-fruit frame in the cell); a fruit not in that frame has no threshold. So: for
# each fruit id of the side-car's fruit supervision, ranked by confirmed 3-D fruit count in the scene graph, pick the
# in-cell keyframe with the most pixels of THAT id and score it (tag "fruitcuts"), then collect the best-IoU threshold per id.
# Output: <side-car dir>/fruit_cuts.json  {"fruit_cuts": {"10001": 0.85, ...}, "detail": {...}}
set -uo pipefail
SV=${1:?survey}; DN=${2:?side-car dir}; CN=${3:?chunk}; MAXF=${4:-6}
S=/home/paperspace/data/citrus_all/$SV; P=${SIDECAR_PROJ:-$S/experimental/h3dgs}; O=$S/experimental/h3dgs_sidecar_chunks/$DN; SUP=$O/supervision/trees_fruit_v3
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=${SIDECAR_KF:-$S/prod/scratch_sam3}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_fruitdensify.log; VL=/home/paperspace/logs/sidecar_${SV}_fruit_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_fruitdensify_r2/high/*/ 2>/dev/null | tail -1)
[ -n "$RUN" ] && [ -e $SUP/manifest.json ] || { say "fruit cuts $DN: no densified run or fruit supervision"; exit 1; }
# fruit ids in this side-car's supervision, ranked by confirmed fruit count (scene graph fruits[i] -> id 10000+i)
IDS=$(python3 - "$HJ" "$SUP/manifest.json" "$MAXF" << 'PY'
import json, sys
h = json.load(open(sys.argv[1])); wt = json.load(open(sys.argv[2]))['word_table']; n = int(sys.argv[3])
have = {int(k) for k in wt if int(k) >= 10000}
ranked = sorted(((10000 + i, f['tree_id'], f.get('n_fruit3d', 0)) for i, f in enumerate(h['fruits']) if 10000 + i in have), key=lambda t: -t[2])
print(' '.join(f'{fid}:{tree}:{n3}' for fid, tree, n3 in ranked[:n]))
PY
)
say "=== fruit cuts $DN: scoring up to $MAXF fruits by confirmed count: $IDS"
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG; cd $NS
for t in $IDS; do
  FID=${t%%:*}; rest=${t#*:}; TREE=${rest%%:*}
  FR=$(python3 /home/paperspace/logs/sidecar_chunk_frame_v2.py $SV $CN --out $O --proj $P --fruit-id $FID 2>>$L | head -1)
  [ -n "$FR" ] || { say "fruit $FID (tree $TREE): no in-cell frame carries it, skipped"; continue; }
  if grep -qE "^\[$DN sidecar (fruitdensify|fruitcuts) $FR\] FRUIT of $TREE " $VL 2>/dev/null; then say "fruit $FID (tree $TREE): frame $FR already scored"; continue; fi
  t0=$(date +%s)
  HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ \
    --supervision-dir $SUP --frame $FR --kf-images $KF --out $FIG/fruitcuts_${DN}_${FID}_$FR 2>&1 \
    | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[$DN sidecar fruitcuts $FR] /" >> $VL
  say "fruit $FID (tree $TREE) frame $FR scored in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[$DN sidecar fruitcuts $FR\] FRUIT of $TREE " $VL | grep -oE 'thr [0-9.]+ IoU [0-9.]+' | head -1)"
  rm -rf $O/clip_cache_*
done
python3 /home/paperspace/logs/sidecar_fruit_cuts.py $SV $DN $O/fruit_cuts.json 2>&1 | tee -a $L
