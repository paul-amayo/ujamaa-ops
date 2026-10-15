#!/bin/bash
# fruit_lod_cell_v6.sh — v5 + SEED=1: mask-centre fruit seeding (fruit_seed_masks at SEED_STEPS, default 3100,15100; scale SEED_FRAC x radius, opacity SEED_OPAC).
# fruit_lod_cell_v5.sh — IDENTITY-DRIVEN LoD (2026-10-14): MIX=1 turns on fruit_mix_lod — trigger = a blob's labelled diet between MIX_LO (0.1) and MIX_HI (0.9) fruit
# (unlabelled px do not vote), action = direct split every 500 steps with no generation cap and a MIX_FLOOR_M (0.01) size floor, children re-judged, split
# scales held, splitting allowed past the 15k window; pure blobs untouched. IT defaults to 30000 with a checkpoint every SAVE_EVERY (15000) steps so the
# 15k and 30k states can be scored (lod_eval_ckpt.sh). The identity / scoring tail of the cell is unchanged and scores the final checkpoint.
# fruit_lod_cell_v3.sh — the FAST mask-size LoD cell (2026-10-13 evening): the v3 recipe (stock splatfacto densification, bilateral grid, 60k) +
# HiGH's fruit tally from a detached re-rasterisation every TALLY_EVERY steps (no feature head: enable-high-features False, 05 1_0 ran 9x slower
# with it) + DIRECT splitting of the protected carriers between FD_SCALE_M and FD_MAX_M (children at scale/1.6, protection inherited; the global
# densify-size-thresh stays the recipe's). Membership = fruit share >= FP_SHARE (0.01 = the measured carriers' share) with tau FP_TAU (3) over
# FP_MINFR (3) sampled frames. TAG=<variant> names the run (high_lod<TAG>_…, identity_lod<TAG>); LOD_W=<W> adds the small-mask L1 weight 1+W
# (HIGH_LOD_LOSS_W); LOD_PX=<px> = general small-mask LoD (every instance <= px); masked PSNR by size scored at the end.
# fruit_lod_cell.sh — recipe v3 + HiGH's fruit-aware densification (Paul 2026-10-13: "we need fruit nerf style densification and splitting …
# opportunistic level of detail on these particular size of masks"): the same workspace, images, poses, 60k iterations, bilateral grid and
# cull as the v3 cell, trained by the HiGH trainer with fruit protection + fruit-targeted splitting (the 1 Oct side-car recipe: gate 4 cm,
# max 0.5 m, split threshold 4 cm, tail 0, all converted through the run's dataparser scale; features head present but its loss warmed-up
# off, as the side-car did) -> census -> seed B -> verdict renders + E1 -> fruit-picked frames scored with the level rule and the mixed rule.
#   WS=<v3 workspace> CELL=<native cell with supervision> N=<v3 identity dir (frames_fruit.json)> fruit_lod_cell.sh   [IT=60000] [FD_SCALE_M=0.04] [FD_MAX_M=0.5] [FD_SPLIT_M=0.04]
# LOD_PX=<px>: mask-size LoD for every instance <= px (HIGH_LOD_MAX_PX), fruit-keyed only when unset.
# FP_TAU (default 30) / FP_MINFR (3): tau 3.0 — the 1,500-step side-car refine's value — protected 558 k of 834 k gaussians by step 3,000 of a 60k run (protection accrues over every epoch) = blanket densification, not LoD
set -uo pipefail; : ${WS:?} ${CELL:?} ${N:?}; IT=${IT:-30000}; SAVE_EVERY=${SAVE_EVERY:-15000}; TAG=${TAG:-}; EXP=high_${LOD_PX:+px${LOD_PX}_}lod${TAG}_${IT}_bilateral; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
S=$(echo $WS | sed -E 's#(.*)/experimental/.*#\1#'); EMB=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])")
HJ=$(python3 -c "import numpy as np, json; print(json.loads(str(np.load('$N/text_bank.npz', allow_pickle=True)['meta']))['hierarchy_json'])")   # the cell's own hierarchy (citrus prod graph, anchor census, image-farm graph)
SUP=$CELL/supervision/trees_only; KF=$S/prod/scratch_sam3; [ -d $KF ] || KF=$WS/images; OUT=$WS/identity_${LOD_PX:+px${LOD_PX}_}lod${TAG}; L=/home/paperspace/logs/lod_cell_$(basename $S)_$(basename $WS)${LOD_PX:+_px$LOD_PX}${TAG}.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(basename $S)/$(basename $WS): $*" | tee -a $L; }; t0=$(date +%s)
DPS=$(python3 -c "import json; print(json.load(open('$(ls -d $WS/outputs/sf_*_bilateral/splatfacto/run | head -1)/dataparser_transforms.json'))['scale'])")
FD="--pipeline.model.fruit-densify-scale $(python3 -c "print(${FD_SCALE_M:-0.02}*$DPS)") --pipeline.model.fruit-densify-max-scale $(python3 -c "print(${FD_MAX_M:-0.5}*$DPS)")"
SEEDF=""; [ "${SEED:-0}" = 1 ] && SEEDF="--pipeline.model.fruit-seed-masks True --pipeline.model.fruit-seed-steps ${SEED_STEPS:-3100,15100} --pipeline.model.fruit-seed-scale-frac ${SEED_FRAC:-0.5} --pipeline.model.fruit-seed-opacity ${SEED_OPAC:-0.5}"
MIXF=""; [ "${MIX:-0}" = 1 ] && MIXF="--pipeline.model.fruit-mix-lod True --pipeline.model.fruit-mix-lo ${MIX_LO:-0.1} --pipeline.model.fruit-mix-hi ${MIX_HI:-0.9} --pipeline.model.fruit-mix-tau ${MIX_TAU:-3} --pipeline.model.fruit-mix-min-size $(python3 -c "print(${MIX_FLOOR_M:-0.01}*$DPS)") --pipeline.model.fruit-mix-until ${MIX_UNTIL:-0} --pipeline.model.fruit-densify-direct-every ${MIX_EVERY:-1000}"
say "=== mask-size LoD (fast): $EXP | seed ${SEED:-0} @${SEED_STEPS:-3100,15100} | mix ${MIX:-0} [${MIX_LO:-0.1},${MIX_HI:-0.9}] tau ${MIX_TAU:-3} every ${MIX_EVERY:-1000} floor ${MIX_FLOOR_M:-0.01} m | densify ${FDENS:-True} | protect tau ${FP_TAU:-3} min-frames ${FP_MINFR:-3} share ${FP_SHARE:-0.01} tally every ${TALLY_EVERY:-4} | L1 weight ${LOD_W:-0} | direct split between ${FD_SCALE_M:-0.02} and ${FD_MAX_M:-0.5} m | dataparser scale $DPS | $FD | sup $SUP | emb $(basename $(dirname $(dirname $EMB)))"
CFG=$WS/outputs/$EXP/high/run/config.yml
if [ ! -e $CFG ]; then t1=$(date +%s)
  (cd $NS && env HIGH_EMBEDDER_CKPT=$EMB HIGH_SEED_DUMP=$WS/outputs/$EXP/fruit_seeds.npz ${LOD_PX:+HIGH_LOD_MAX_PX=$LOD_PX} ${LOD_W:+HIGH_LOD_LOSS_W=$LOD_W} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 pixi run ns-train high --data $WS --output-dir $WS/outputs --experiment-name $EXP --timestamp run --vis tensorboard \
     --max-num-iterations $IT --steps-per-save $SAVE_EVERY --save-only-latest-checkpoint False --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 --pipeline.model.use-bilateral-grid True \
     --pipeline.model.enable-high-features False --pipeline.model.fruit-protect True --pipeline.model.fruit-tally-every ${TALLY_EVERY:-4} --pipeline.model.fruit-protect-tau ${FP_TAU:-3} --pipeline.model.fruit-protect-min-frames ${FP_MINFR:-3} --pipeline.model.fruit-protect-share ${FP_SHARE:-0.01} --pipeline.model.fruit-anchor-weight 0.0 \
     --pipeline.model.fruit-densify ${FDENS:-True} --pipeline.model.fruit-densify-direct-split True --pipeline.model.fruit-densify-tail 0 --pipeline.model.stop-split-at 15000 $FD $MIXF $SEEDF --pipeline.datamanager.semantic-dir $SUP \
     nerfstudio-data --eval-mode filename) > $L.train 2>&1
  say "train rc=$? in $(( ($(date +%s)-t1)/60 )) min: $(grep -a -c '\[fruit-densify\] step' $L.train) split lines, lod-loss lines $(grep -a -c '\[lod-loss\]' $L.train), last: $(grep -a '\[fruit-densify\] step' $L.train | tail -1 | tr -s ' ' | cut -c1-120); protect: $(grep -a '\[fruit-protect\]' $L.train | tail -1 | tr -s ' ' | cut -c1-100)"
fi
[ -e $CFG ] || { say "no config — training FAILED (see $L.train)"; exit 1; }
python3 - "$CFG" <<'PY' | tee -a $L
import sys, torch, glob, os
ck = sorted(glob.glob(os.path.join(os.path.dirname(sys.argv[1]), 'nerfstudio_models', '*.ckpt')))[-1]; c = torch.load(ck, map_location='cpu', weights_only=False)
n = [v for k, v in c['pipeline'].items() if k.endswith('gauss_params.means')][0].shape[0]; print(f'[fruit-lod] final checkpoint {os.path.basename(ck)}: {n} gaussians')
PY
[ -e $OUT/verdicts.log ] || CFG=$CFG CELL=$CELL EMB=$EMB HJ=$HJ KF=$KF OUT=$OUT SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh 2>&1 | grep -aE "v3 means|census|assigned|done in" | tee -a $L
cp -n $N/frames_fruit.json $OUT/ 2>/dev/null; FR=$(python3 -c "import json; print(' '.join(json.load(open('$OUT/frames_fruit.json'))['frames']))"); SP=$WS/sparse/0; [ -e $SP/images.bin ] || SP=$(dirname $N)/sparse/0
[ -e $OUT/maps_fruit/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || (cd $NS && nice -n 10 pixi run python $V3/splat_feature_render.py --config $CFG --features $OUT/features_B.bin --frames $FR --out-dir $OUT/maps_fruit --colmap-sparse $SP 2>&1) | grep -aE "render\] splatfacto|Error|Traceback" | tee -a $L
for rule in level mixed; do [ -s $OUT/protocol_lod${TAG}_fruit_$rule.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag lod${TAG}_fruit_$rule --space hyper --maps maps_fruit/%s.npz --frames-json $OUT/frames_fruit.json --fruit-rule $rule 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | cut -c1-400 | tee -a $L; done
[ -s $OUT/protocol_lod${TAG}.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag lod${TAG} --space hyper 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | cut -c1-300 | tee -a $L
# rendering quality inside the masks by object size — the LoD claim itself (same picks as the stock numbers: 30 census views + every held-out view; and the fruit-picked frames)
[ -s $OUT/masked_psnr.json ] || (cd $NS && nice -n 10 pixi run python $V3/masked_psnr_by_size.py --config $CFG --sup $SUP --out $OUT/masked_psnr.json 2>&1) | grep -aE "^\[masked-psnr\]|Error|Traceback" | tee -a $L
[ -s $OUT/masked_psnr_fruitframes.json ] || (cd $NS && nice -n 10 pixi run python $V3/masked_psnr_by_size.py --config $CFG --sup $SUP --frames $FR --out $OUT/masked_psnr_fruitframes.json 2>&1) | grep -aE "^\[masked-psnr\]|Error|Traceback" | tee -a $L
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
