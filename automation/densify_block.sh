#!/bin/bash
# Mask-driven fruit densification pass (2026-08-26, Paul's design).
#
#   usage: densify_block.sh <block_dir> <supervision_dir> [iters]
#
# WHY: the census argmax hands a fruit's own pixels to the tree — measured on 05
# b000, fruit-assigned gaussians hold 0-10% of the blend AT the fruit's pixels,
# because every gaussian large enough to render fruit also renders thousands of
# tree px. No amount of feature training fixes a blend the fruit doesn't own;
# the fix is GEOMETRIC: split the coarse carriers until small, fruit-dieted
# gaussians exist (diet>50% share was 8-23% pre-densify on b000).
#
# HOW (the cheap regime — NOT the 20k s9 run):
#   - resume from the zero-feature stage2_init ckpt, features ENABLED so the
#     fruit-protect tally can probe the rendered feature map (the probes measure
#     pure blend weight on a zero field — the tally IS the diet),
#   - HIGH_LOSS_WARMUP_STEP=10^9 zeroes the feature loss while keeping the
#     graph: features never train, so the LR schedule is irrelevant (the whole
#     point), and fruit-anchor-weight 0 removes the only other feature loss,
#   - fruit-protect + fruit-densify: mask-tallied protection boosts coarse
#     (>4 cm) carriers past the grow threshold each refine -> gsplat splits
#     them; self-limiting at the scale gate; children inherit protection,
#   - geometry live but at the resumed schedule's tail LR, so nothing drifts —
#     the only intended change is the split topology.
#
# After this pass, re-census: argmax and diet now agree, the seed init lands on
# the gaussians that own the fruit pixels, and seed-only economics hold.
set -u
BD=$(readlink -f "$1"); SUP=$(readlink -f "$2"); N=$(basename "$BD")
ITERS=${3:-2000}
export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
EMB=${CENSUS_EMBEDDER:?set CENSUS_EMBEDDER}

S1=$(ls -t "$BD"/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt | head -1)
STEP0=$(basename "$S1" | grep -oE '[0-9]+' | sed 's/^0*//')
MAXIT=$((STEP0 + ITERS + 1))

# stage2_init (stage1 + zero high_features) — reclaimed after verdicts, rebuild
if [ ! -f "$BD/stage2_init/nerfstudio_models/$(basename "$S1")" ]; then
  (cd /home/paperspace/code/nerf_new && pixi run python - "$S1" "$BD/stage2_init/nerfstudio_models" << 'PY'
import sys, torch
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location='cpu')
means_key = [k for k in ck['pipeline'] if k.endswith('gauss_params.means')][0]
n = ck['pipeline'][means_key].shape[0]
hf_key = means_key.replace('means', 'high_features')
assert hf_key not in ck['pipeline']
ck['pipeline'][hf_key] = torch.zeros((n, 32))
dst.mkdir(parents=True, exist_ok=True)
torch.save(ck, dst / src.name)
print(f'DENSIFY-INIT0: zero-feature init ({n} gaussians)')
PY
  ) || { echo "DENSIFY-FAIL: init0"; exit 1; }
fi

echo "DENSIFY: resume step $STEP0 -> $MAXIT ($ITERS iters), feature loss ZEROED, splits on"
cd /home/paperspace/code/nerf_new
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB HIGH_LOSS_WARMUP_STEP=1000000000 \
  pixi run ns-train high \
    --data "$BD" --output-dir "$BD/splat_runs_FEATFIX" --experiment-name fruit_densify \
    --load-dir "$BD/stage2_init/nerfstudio_models" \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.model.fruit-protect-tau 3.0 \
    --pipeline.model.fruit-anchor-weight 0.0 \
    --pipeline.model.fruit-densify True \
    --pipeline.model.stop-split-at $MAXIT \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.datamanager.semantic-dir "$SUP" \
    --max-num-iterations $MAXIT --steps-per-save $((MAXIT - 1)) \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 \
    || { echo "DENSIFY-FAIL: train"; exit 1; }

CK=$(ls -t "$BD"/splat_runs_FEATFIX/fruit_densify/high/*/nerfstudio_models*/*.ckpt | head -1)
[ -n "$CK" ] || { echo "DENSIFY-FAIL: no ckpt"; exit 1; }
echo "DENSIFY-DONE $N -> $CK"
