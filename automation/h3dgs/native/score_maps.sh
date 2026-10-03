#!/bin/bash
# score_maps.sh <cell dir N> <model tag> <npz suffix> <supervision dir> <embedder> <hierarchy json> <kf images> [parallel=4]
# Scores <N>/<maps dir>/<frame stem><suffix>.npz for every frame in <N>/frames.json with containment_eval.py --features-npz
# and appends "[<tag> <frame>] TREE ..." lines to <N>/verdicts.log (UJAMAA, 2026-10-03).
#   e.g. score_maps.sh $N native_tau9 maps_native/%s_tau9.npz $SUP $EMB $HJ $KF
set -uo pipefail
N=$1; TAG=$2; PAT=$3; SUP=$4; EMB=$5; HJ=$6; KF=$7; PAR=${8:-4}
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new; mkdir -p $N/figs
one(){ f=$1; z=$N/$(printf "$PAT" "${f%.png}"); [ -e "$z" ] || { echo "[$TAG $f] MISSING $z"; return; }
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/containment_eval.py --features-npz $z --hyper-ckpt $EMB --hierarchy-json $HJ \
     --supervision-dir $SUP --frame $f --kf-images $KF --out $N/figs/${f%.png}_$TAG.png 2>&1) | grep -aE '^(TREE|ROW|FRUIT) |norm gate' | sed "s/^/[$TAG $f] /"; }
export -f one; export N TAG PAT SUP EMB HJ KF ARU NS
python3 -c "import json; print('\n'.join(json.load(open('$N/frames.json'))['frames']))" | xargs -P $PAR -I{} bash -c 'one {}' >> $N/verdicts.log
echo "[score_maps] $TAG: $(grep -c "^\[$TAG " $N/verdicts.log) lines in $N/verdicts.log"
