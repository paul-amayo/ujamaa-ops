#!/bin/bash
# fruit_lod_cell_v2.sh — TAG=<variant> names the run (high_fd<TAG>_…, identity_fd<TAG>), LOD_W=<W> adds the small-mask L1 weight 1+W (HIGH_LOD_LOSS_W), masked PSNR by size scored at the end.
# fruit_lod_cell.sh — recipe v3 + HiGH's fruit-aware densification (Paul 2026-10-13: "we need fruit nerf style densification and splitting …
# opportunistic level of detail on these particular size of masks"): the same workspace, images, poses, 60k iterations, bilateral grid and
# cull as the v3 cell, trained by the HiGH trainer with fruit protection + fruit-targeted splitting (the 1 Oct side-car recipe: gate 4 cm,
# max 0.5 m, split threshold 4 cm, tail 0, all converted through the run's dataparser scale; features head present but its loss warmed-up
# off, as the side-car did) -> census -> seed B -> verdict renders + E1 -> fruit-picked frames scored with the level rule and the mixed rule.
#   WS=<v3 workspace> CELL=<native cell with supervision> N=<v3 identity dir (frames_fruit.json)> fruit_lod_cell.sh   [IT=60000] [FD_SCALE_M=0.04] [FD_MAX_M=0.5] [FD_SPLIT_M=0.04]
# LOD_PX=<px>: mask-size LoD for every instance <= px (HIGH_LOD_MAX_PX), fruit-keyed only when unset.
# FP_TAU (default 30) / FP_MINFR (3): tau 3.0 — the 1,500-step side-car refine's value — protected 558 k of 834 k gaussians by step 3,000 of a 60k run (protection accrues over every epoch) = blanket densification, not LoD
set -uo pipefail; : ${WS:?} ${CELL:?} ${N:?}; IT=${IT:-60000}; TAG=${TAG:-}; EXP=high_${LOD_PX:+lod${LOD_PX}_}fd${TAG}_${IT}_bilateral; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new
S=$(echo $WS | sed -E 's#(.*)/experimental/.*#\1#'); EMB=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])")
HJ=$(python3 -c "import numpy as np, json; print(json.loads(str(np.load('$N/text_bank.npz', allow_pickle=True)['meta']))['hierarchy_json'])")   # the cell's own hierarchy (citrus prod graph, anchor census, image-farm graph)
SUP=$CELL/supervision/trees_only; KF=$S/prod/scratch_sam3; [ -d $KF ] || KF=$WS/images; OUT=$WS/identity_${LOD_PX:+lod${LOD_PX}_}fd${TAG}; L=/home/paperspace/logs/fruit_lod_$(basename $S)_$(basename $WS)${LOD_PX:+_lod$LOD_PX}${TAG}.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(basename $S)/$(basename $WS): $*" | tee -a $L; }; t0=$(date +%s)
DPS=$(python3 -c "import json; print(json.load(open('$(ls -d $WS/outputs/sf_*_bilateral/splatfacto/run | head -1)/dataparser_transforms.json'))['scale'])")
FD="--pipeline.model.fruit-densify-scale $(python3 -c "print(${FD_SCALE_M:-0.04}*$DPS)") --pipeline.model.fruit-densify-max-scale $(python3 -c "print(${FD_MAX_M:-0.5}*$DPS)") --pipeline.model.densify-size-thresh $(python3 -c "print(${FD_SPLIT_M:-0.04}*$DPS)")"
say "=== fruit LoD: $EXP | protect tau ${FP_TAU:-30} min-frames ${FP_MINFR:-3} share ${FP_SHARE:-0.3} | L1 weight ${LOD_W:-0} | split gate ${FD_SCALE_M:-0.04} m / thresh ${FD_SPLIT_M:-0.04} m | dataparser scale $DPS | $FD | sup $SUP | emb $(basename $(dirname $(dirname $EMB)))"
CFG=$WS/outputs/$EXP/high/run/config.yml
if [ ! -e $CFG ]; then t1=$(date +%s)
  (cd $NS && env HIGH_EMBEDDER_CKPT=$EMB HIGH_LOSS_WARMUP_STEP=1000000000 ${LOD_PX:+HIGH_LOD_MAX_PX=$LOD_PX} ${LOD_W:+HIGH_LOD_LOSS_W=$LOD_W} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 pixi run ns-train high --data $WS --output-dir $WS/outputs --experiment-name $EXP --timestamp run --vis tensorboard \
     --max-num-iterations $IT --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 --pipeline.model.use-bilateral-grid True \
     --pipeline.model.high-loss-weight 1.0 --pipeline.model.fruit-protect True --pipeline.model.fruit-protect-tau ${FP_TAU:-30} --pipeline.model.fruit-protect-min-frames ${FP_MINFR:-3} --pipeline.model.fruit-protect-share ${FP_SHARE:-0.3} --pipeline.model.fruit-anchor-weight 0.0 \
     --pipeline.model.fruit-densify True --pipeline.model.fruit-densify-tail 0 --pipeline.model.stop-split-at 15000 $FD --pipeline.datamanager.semantic-dir $SUP \
     nerfstudio-data --eval-mode filename) > $L.train 2>&1
  say "train rc=$? in $(( ($(date +%s)-t1)/60 )) min: $(grep -a -c '\[fruit-densify\] step' $L.train) boost lines, lod-loss lines $(grep -a -c '\[lod-loss\]' $L.train), last: $(grep -a '\[fruit-densify\] step' $L.train | tail -1 | tr -s ' ' | cut -c1-120); protect: $(grep -a '\[fruit-protect\]' $L.train | tail -1 | tr -s ' ' | cut -c1-100)"
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
for rule in level mixed; do [ -s $OUT/protocol_fd${TAG}_fruit_$rule.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag fd${TAG}_fruit_$rule --space hyper --maps maps_fruit/%s.npz --frames-json $OUT/frames_fruit.json --fruit-rule $rule 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | cut -c1-400 | tee -a $L; done
[ -s $OUT/protocol_fd${TAG}.jsonl ] || (cd $NS && nice -n 10 pixi run python $V3/argmax_eval.py --identity $OUT --tag fd${TAG} --space hyper 2>&1) | grep -aE "^\[protocol\]|Error|Traceback" | cut -c1-300 | tee -a $L
# rendering quality inside the masks by object size — the LoD claim itself (same picks as the stock numbers: 30 census views + every held-out view; and the fruit-picked frames)
[ -s $OUT/masked_psnr.json ] || (cd $NS && nice -n 10 pixi run python $V3/masked_psnr_by_size.py --config $CFG --sup $SUP --out $OUT/masked_psnr.json 2>&1) | grep -aE "^\[masked-psnr\]|Error|Traceback" | tee -a $L
[ -s $OUT/masked_psnr_fruitframes.json ] || (cd $NS && nice -n 10 pixi run python $V3/masked_psnr_by_size.py --config $CFG --sup $SUP --frames $FR --out $OUT/masked_psnr_fruitframes.json 2>&1) | grep -aE "^\[masked-psnr\]|Error|Traceback" | tee -a $L
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
