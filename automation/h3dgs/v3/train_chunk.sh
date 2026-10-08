#!/bin/bash
# Recipe v3 chunk training (2026-10-08): stock nerfstudio splatfacto + bilateral grid on a v3 workspace (build_chunk_ws.py), the
# configuration the 05 1_0 A/B measured (60k, cull-alpha 0.005, eval-mode filename), then the two scores that A/B used: held-out raw
# (ns-eval, nerfstudio PSNR) and the 60 training views WITH the trained grid (splat_bilateral_score.py). nerf_new pixi env.
#   train_chunk.sh <workspace> [iters=60000] [run name=sf_<iters>_bilateral]
set -u; WS=$1; IT=${2:-60000}; N=${3:-sf_${IT}_bilateral}; NS=/home/paperspace/code/nerf_new
TAG=${TAG:-$(python3 -c "import json; s=json.load(open('$WS/split.json')); print(s['project'].split('/')[5] + '_' + s['chunk'])")}
L=/home/paperspace/logs/v3_$TAG.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; cd $NS; t0=$(date +%s)
say "=== v3 $N on $WS ($(python3 -c "import json; s=json.load(open('$WS/split.json')); print(len(s['train']), 'train /', len(s['test']), 'held-out')"))"
pixi run ns-train splatfacto --data $WS --output-dir $WS/outputs --experiment-name $N --timestamp run --vis tensorboard --max-num-iterations $IT \
  --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.use-bilateral-grid True nerfstudio-data --eval-mode filename > /home/paperspace/logs/v3_${TAG}_${N}_train.log 2>&1
say "$N train rc=$? in $(( $(date +%s)-t0 )) s"
CFG=$WS/outputs/$N/splatfacto/run/config.yml; [ -e $CFG ] || { say "$N: no config.yml — see v3_${TAG}_${N}_train.log"; exit 1; }
pixi run ns-eval --load-config $CFG --output-path $WS/outputs/$N/eval_heldout.json > /home/paperspace/logs/v3_${TAG}_${N}_eval.log 2>&1
say "$N held-out raw: $(python3 -c "import json; d=json.load(open('$WS/outputs/$N/eval_heldout.json'))['results']; print('psnr %.2f ssim %.3f lpips %.3f' % (d['psnr'], d['ssim'], d['lpips']))" 2>&1)"
pixi run python /home/paperspace/code/automation/h3dgs/fruit_field/splat_bilateral_score.py $CFG 2>&1 | grep -a '^\[bilateral\]' | tee -a $L
say "=== v3 $N done in $(( ($(date +%s)-t0)/60 )) min; gaussians $(pixi run python -c "import torch,glob; c=torch.load(sorted(glob.glob('$WS/outputs/$N/splatfacto/run/nerfstudio_models/*.ckpt'))[-1], map_location='cpu'); print(c['pipeline']['_model.gauss_params.means'].shape[0])" 2>/dev/null)"
