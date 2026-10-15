#!/bin/bash
# lod_majority_test.sh — re-seed an identity with a stricter fruit rule: SHARE=<fruit-share-assign> (default 0 = MAJORITY: a gaussian gets the fruit word only where fruit
# wins its diet, like trees) instead of the 10 % share rule, on the SAME geometry; render the fruit frames, score E4 (level + mixed) and draw the
# identity figure 10 %-rule vs majority (Paul 2026-10-13: "why are we using identity for fruit gaussians that do not touch fruit, surely we will
# always leak this way?").   CFG=<run config.yml> SRC=<identity dir (W.npz, text_bank, frames_fruit.json)> DST=<new identity dir> TAG=<tag> lod_majority_test.sh
set -uo pipefail; : ${CFG:?} ${SRC:?} ${DST:?} ${TAG:?}; SHARE=${SHARE:-0}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; ARU=/home/paperspace/code/aru_sil_core/src/scripts
WS=$(dirname $(dirname $(dirname $(dirname $(dirname $CFG)))));   # <ws>/outputs/<exp>/<method>/run/config.yml -> <ws> (sparse/0 lives there) SUP=${SUP:-/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only}
EMB=$(python3 -c "import json; print(json.load(open('$SRC/features_B.bin.json'))['embedder'])"); mkdir -p $DST; for f in W.npz W.json text_bank.npz frames.json frames_fruit.json split_names.json verdicts.log; do [ -e $SRC/$f ] && cp -n $SRC/$f $DST/; done
L=/home/paperspace/logs/lod_majority_$TAG.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $TAG: $*" | tee -a $L; }
say "=== re-seed with fruit-share-assign $SHARE (0 = majority) of $SRC -> $DST (geometry $CFG)"
[ -s $DST/features_B.bin ] || (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $DST/W.npz --embedder $EMB --out-features $DST/features_B.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign $SHARE 2>&1) | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
python3 - $DST <<'PY' | tee -a $L
import numpy as np, sys; l=np.load(sys.argv[1]+'/features_B.bin.labels.npy'); print(f'[majority] labels: {int((l>=0).sum())} assigned, {int(((l>=10000)&(l<65535)).sum())} fruit-word gaussians')
PY
FR=$(python3 -c "import json; print(' '.join(json.load(open('$DST/frames_fruit.json'))['frames']))")
[ -e $DST/maps_fruit/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || (cd $NS && nice -n 10 pixi run python $V3/splat_feature_render.py --config $CFG --features $DST/features_B.bin --frames $FR --out-dir $DST/maps_fruit --colmap-sparse $WS/sparse/0 2>&1) | grep -aE "render\] splatfacto|render\] high|Error|Traceback" | tee -a $L
for rule in level mixed; do [ -s $DST/protocol_${TAG}_fruit_$rule.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $DST --tag ${TAG}_fruit_$rule --space hyper --maps maps_fruit/%s.npz --frames-json $DST/frames_fruit.json --fruit-rule $rule 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | cut -c1-400 | tee -a $L; done
FIGDIR=${FIGDIR:-/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity/figs/lod}; mkdir -p $FIGDIR
(cd $NS && nice -n 10 pixi run python $V3/lod_identity_fig.py --before $CFG:$SRC --after $CFG:$DST --sup $SUP --frames-json $DST/frames_fruit.json --out $FIGDIR/lod_identity_${TAG}.png 2>&1) | grep -aE "^\[lod-id\]|Error|Traceback" | cut -c1-1200 | tee -a $L
say "=== done"
