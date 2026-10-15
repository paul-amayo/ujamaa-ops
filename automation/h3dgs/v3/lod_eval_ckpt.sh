#!/bin/bash
# lod_eval_ckpt.sh — score ONE checkpoint of a run under the identity-LoD metrics (2026-10-14): census (labelled votes kept separate from the unlabelled
# row) -> two seeds (today's 10 % share rule; labelled-majority 0.5 = fruit if >= half its labelled votes, unlabelled px do not vote) -> fruit-frame maps
# -> E4 level rule for both -> purity (share of the oranges' rendered mass from blobs >= 50 % fruit) -> masked PSNR on the fruit frames -> per-pixel
# predictions on FIGFRAME for the rule figure.   CFG=<run config.yml> [STEP=14999] N=<reference identity dir: split_names, frames_fruit, text_bank>
#   SUP=<id maps> OUT=<identity dir to create> TAG=<tag> [FIGFRAME=kf_000025.png] lod_eval_ckpt.sh
set -uo pipefail; : ${CFG:?} ${N:?} ${SUP:?} ${OUT:?} ${TAG:?}; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; ARU=/home/paperspace/code/aru_sil_core/src/scripts
RUN=$(dirname $CFG); WS=$(dirname $(dirname $(dirname $(dirname $RUN)))); L=/home/paperspace/logs/lod_eval_$TAG.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $TAG: $*" | tee -a $L; }; t0=$(date +%s)
if [ -n "${STEP:-}" ]; then CK=$(ls $RUN/nerfstudio_models/ | grep -E "step-0*${STEP}\.ckpt" | head -1); [ -n "$CK" ] || { say "no checkpoint for step $STEP in $RUN/nerfstudio_models"; exit 1; }
  S=$(echo $CK | sed -E 's/step-0*([0-9]+)\.ckpt/\1/'); CFGS=$RUN/config_step$S.yml; [ -e $CFGS ] || sed -E "s/^load_step: .*/load_step: $S/" $CFG > $CFGS; grep -q "^load_step: $S" $CFGS || { say "config has no load_step line"; exit 1; }; else CFGS=$CFG; fi
EMB=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])"); mkdir -p $OUT; for f in text_bank.npz frames.json frames_fruit.json split_names.json verdicts.log; do [ -e $N/$f ] && cp -n $N/$f $OUT/; done
FR=$(python3 -c "import json; print(' '.join(json.load(open('$OUT/frames_fruit.json'))['frames']))"); FRT=$(python3 -c "import json; d=json.load(open('$OUT/frames_fruit.json')); print(' '.join(f for f in d['frames'] if d['info'].get(f,{}).get('split')=='train'))")
say "=== eval $CFGS -> $OUT"
[ -s $OUT/W2.npz ] || { rm -f $OUT/W.npz; (cd $NS && nice -n 5 pixi run python $V3/splat_census.py --config $CFGS --supervision-dir $SUP --names-json $OUT/split_names.json --with-bg --fruit-views-npz $OUT/W2.npz --out-npz $OUT/W.npz 2>&1) | grep -aE '^\[census\] saved|^\[census\] fruit-views|Error|Traceback' | cut -c1-200 | tee -a $L; }
for rule in 0.1 0.5; do t=$(echo $rule | tr -d .); F=$OUT/features_s$t.bin
  [ -s $F ] || (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $OUT/W.npz --embedder $EMB --out-features $F --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign $rule 2>&1) | grep -aE 'assigned|Error|Traceback' | tee -a $L
  [ -e $OUT/features_B.bin.json ] || cp $F.json $OUT/features_B.bin.json
  [ -e $OUT/maps_s$t/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || (cd $NS && nice -n 10 pixi run python $V3/splat_feature_render.py --config $CFGS --features $F --frames $FR --out-dir $OUT/maps_s$t --colmap-sparse $WS/sparse/0 2>&1) | grep -aE "Error|Traceback" | tee -a $L
  [ -s $OUT/protocol_e4_s${t}_fruit_level.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag e4_s${t}_fruit_level --space hyper --maps maps_s$t/%s.npz --frames-json $OUT/frames_fruit.json --fruit-rule level --dump-pred $OUT/pred_s$t 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | sed -E 's/.*\[hyper\]: //' | cut -c1-330 | sed "s/^/[e4 rule $rule] /" | tee -a $L
done
# mask-views fruit rule: trees from the 0.5 seed's base assignment, fruit judged only on the views that label it (W2), then re-seeded
t=mv; F=$OUT/features_mv.bin
[ -s $OUT/labels_mv.npy ] || python3 $V3/fruit_relabel_views.py --base $OUT/features_s05.bin.labels.npy --w-npz $OUT/W.npz --w2-npz $OUT/W2.npz --out $OUT/labels_mv.npy 2>&1 | tee -a $L
[ -s $F ] || (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $V3/reseed_from_labels.py --labels $OUT/labels_mv.npy --embedder $EMB --out $F 2>&1) | grep -aE "reseed|Error|Traceback" | cut -c1-200 | tee -a $L
[ -e $OUT/maps_mv/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || (cd $NS && nice -n 10 pixi run python $V3/splat_feature_render.py --config $CFGS --features $F --frames $FR --out-dir $OUT/maps_mv --colmap-sparse $WS/sparse/0 2>&1) | grep -aE "Error|Traceback" | tee -a $L
[ -s $OUT/protocol_e4_mv_fruit_level.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag e4_mv_fruit_level --space hyper --maps maps_mv/%s.npz --frames-json $OUT/frames_fruit.json --fruit-rule level --dump-pred $OUT/pred_mv 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | sed -E 's/.*\[hyper\]: //' | cut -c1-330 | sed "s/^/[e4 rule mask-views] /" | tee -a $L
[ -s $OUT/purity.json ] || (cd $NS && nice -n 10 pixi run python $V3/fruit_purity.py --config $CFGS --w-npz $OUT/W.npz --sup $SUP --frames $FRT --out $OUT/purity.json 2>&1) | grep -aE "^\[purity\]|Error|Traceback" | cut -c1-500 | tee -a $L
[ -s $OUT/masked_psnr_fruitframes.json ] || (cd $NS && nice -n 10 pixi run python $V3/masked_psnr_by_size.py --config $CFGS --sup $SUP --frames $FR --out $OUT/masked_psnr_fruitframes.json 2>&1) | grep -aE "^\[masked-psnr\]|Error|Traceback" | cut -c1-300 | tee -a $L
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
