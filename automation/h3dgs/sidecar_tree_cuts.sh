#!/bin/bash
# sidecar_tree_cuts.sh <survey> <side-car dir name> <chunk> [max trees=10] — per-TREE containment verdicts on a chunk side-car,
# then the fitted thresholds as JSON for the app (dashboard session, 2026-09-30: with the per-frame Otsu split a tree query
# at 8 m also lit the neighbouring canopy; the app passes the fitted threshold as the query's `cut`, as it does for fruit).
# The side-car's own verdict scores ONE frame; here the top trees by supervised pixels over the IN-CELL keyframes each get
# the in-cell frame with the most pixels of that id (tag "treecuts"), and sidecar_tree_cuts.py collects the best-IoU threshold
# per id over every scored frame. Output: <side-car dir>/tree_cuts.json  {"tree_cuts": {"90": 0.88, ...}, "detail": {...}}
set -uo pipefail
SV=${1:?survey}; DN=${2:?side-car dir}; CN=${3:?chunk}; MAXT=${4:-10}
S=/home/paperspace/data/citrus_all/$SV; P=${SIDECAR_PROJ:-$S/experimental/h3dgs}; O=$S/experimental/h3dgs_sidecar_chunks/$DN; SUP=$O/supervision/trees_only
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=${SIDECAR_KF:-$S/prod/scratch_sam3}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/sidecar_${SV}_chunks.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ 2>/dev/null | tail -1)
[ -n "$RUN" ] && [ -e $SUP/manifest.json ] || { say "tree cuts $DN: no seed run or trees_only/manifest.json"; exit 1; }
# rank tree ids by supervised pixels over the in-cell keyframes, and the frame each id is largest in
RANK=$(python3 - "$P" "$CN" "$O" "$MAXT" << 'PY'
import json, sys, numpy as np
from pathlib import Path
from PIL import Image
P, cn, O, n = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], float)[:3, :3]
c = np.loadtxt(P / f'camera_calibration/chunks/{cn}/center.txt'); e = np.loadtxt(P / f'camera_calibration/chunks/{cn}/extent.txt'); lo, hi = c - e / 2, c + e / 2
inside = set()
for f in json.load(open(O / 'transforms.json'))['frames']:
    p = R_W @ np.asarray(f['transform_matrix'], float)[:3, 3]
    if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]: inside.add(Path(f['file_path']).name)
tot = {}; bestf = {}
for f in sorted((O / 'supervision/trees_only').glob('kf_*.png')):
    if f.name not in inside: continue
    m = np.array(Image.open(f), np.uint16); ids, cnt = np.unique(m[(m != 65535) & (m < 10000)], return_counts=True)
    for i, k in zip(ids.tolist(), cnt.tolist()):
        tot[i] = tot.get(i, 0) + k
        if k > bestf.get(i, (0, ''))[0]: bestf[i] = (k, f.name)
ranked = sorted(tot.items(), key=lambda t: -t[1])[:n]
print(' '.join(f'{i}:{bestf[i][1]}:{tot_}' for i, tot_ in ranked))
PY
)
say "=== tree cuts $DN: top $MAXT trees by in-cell supervised pixels: $(echo $RANK | tr ' ' '\n' | cut -d: -f1 | paste -sd,)"
FIG=/home/paperspace/logs/sidecar_figs_$SV; mkdir -p $FIG; cd $NS
for t in $RANK; do
  TID=${t%%:*}; rest=${t#*:}; FR=${rest%%:*}
  if grep -qE "^\[$DN sidecar [a-z0-9_.]+ $FR\] TREE $TID " $VL 2>/dev/null; then say "tree $TID: frame $FR already scored"; continue; fi
  t0=$(date +%s)
  HIGH_EMBEDDER_CKPT=$EMB timeout 1800 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ \
    --supervision-dir $SUP --frame $FR --kf-images $KF --out $FIG/treecuts_${DN}_${TID}_$FR 2>&1 \
    | grep -aE "^(TREE|ROW|FRUIT) " | sed "s/^/[$DN sidecar treecuts $FR] /" >> $VL
  say "tree $TID frame $FR scored in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[$DN sidecar treecuts $FR\] TREE $TID " $VL | grep -oE 'thr [0-9.]+ IoU [0-9.]+' | head -1)"
  rm -rf $O/clip_cache_*
done
python3 /home/paperspace/logs/sidecar_tree_cuts.py $SV $DN $O/tree_cuts.json 2>&1 | tee -a $L
