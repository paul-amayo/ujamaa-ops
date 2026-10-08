#!/bin/bash
# train_chunk_x.sh — train_chunk.sh with recipe knobs for the v3 recipe check (2026-10-08, after 05 0_0 and 01 3_1 scored 1.0-1.3 dB
# under the v2 bar on training views): EXTRA = extra ns-train arguments (model/trainer knobs, before `nerfstudio-data`), the run name
# is the third argument. Same scores as train_chunk.sh (held-out raw via ns-eval, 60 training views WITH the grid).
#   EXTRA="--pipeline.model.densify-grad-thresh 0.0004" TAG=05_0_0_cap train_chunk_x.sh <workspace> <iters> <run name>
set -u; WS=$1; IT=${2:-60000}; N=${3:-sf_${IT}_bilateral}; NS=/home/paperspace/code/nerf_new; EXTRA=${EXTRA:-}
TAG=${TAG:-$(python3 -c "import json; s=json.load(open('$WS/split.json')); print(s['project'].split('/')[5] + '_' + s['chunk'])")}
L=/home/paperspace/logs/v3_$TAG.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; cd $NS; t0=$(date +%s)
say "=== v3 $N on $WS ($(python3 -c "import json; s=json.load(open('$WS/split.json')); print(len(s['train']), 'train /', len(s['test']), 'held-out')")) iters $IT extra: ${EXTRA:-none}"
pixi run ns-train splatfacto --data $WS --output-dir $WS/outputs --experiment-name $N --timestamp run --vis tensorboard --max-num-iterations $IT \
  --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.use-bilateral-grid True $EXTRA nerfstudio-data --eval-mode filename > /home/paperspace/logs/v3_${TAG}_${N}_train.log 2>&1
say "$N train rc=$? in $(( $(date +%s)-t0 )) s"
CFG=$WS/outputs/$N/splatfacto/run/config.yml; [ -e $CFG ] || { say "$N: no config.yml — see v3_${TAG}_${N}_train.log"; exit 1; }
pixi run ns-eval --load-config $CFG --output-path $WS/outputs/$N/eval_heldout.json > /home/paperspace/logs/v3_${TAG}_${N}_eval.log 2>&1
say "$N held-out raw: $(python3 -c "import json; d=json.load(open('$WS/outputs/$N/eval_heldout.json'))['results']; print('psnr %.2f ssim %.3f lpips %.3f' % (d['psnr'], d['ssim'], d['lpips']))" 2>&1)"
pixi run python /home/paperspace/code/automation/h3dgs/fruit_field/splat_bilateral_score.py $CFG 2>&1 | grep -a '^\[bilateral\]' | tee -a $L
say "=== v3 $N done in $(( ($(date +%s)-t0)/60 )) min; gaussians $(pixi run python -c "import torch,glob; c=torch.load(sorted(glob.glob('$WS/outputs/$N/splatfacto/run/nerfstudio_models/*.ckpt'))[-1], map_location='cpu'); print(c['pipeline']['_model.gauss_params.means'].shape[0])" 2>/dev/null)"
