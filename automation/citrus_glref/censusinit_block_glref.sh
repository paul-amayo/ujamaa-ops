#!/bin/bash
# CENSUS-INIT stage-2 SEED for a glref block — the recipe of record (automation/censusinit_block.sh,
# 2026-08-07) with every input/output name parametrised so the corrected era lives BESIDE the
# served prod runs (prod doctrine: never overwrite/delete):
#   stage1 run  : splat_runs_STAGE1/stage1_bg00_glref            (fleet output, GL+refined poses)
#   outputs     : stage2_init_glref/, splat_runs_FEATFIX/stage2_bootstrap_glref/, interaction_W_glref.npz,
#                 stage2_init_census_glref/, splat_runs_FEATFIX/stage2_censusinit_glref/high/<ts>/
# usage: censusinit_block_glref.sh <block_dir>   (env: CENSUS_EMBEDDER, CENSUS_HIERARCHY)
set -uo pipefail
BD=$(readlink -f "$1"); N=$(basename "$BD"); SUP=$BD/supervision/trees_only
EMB=${CENSUS_EMBEDDER:?}; HJ=${CENSUS_HIERARCHY:?}
ARU=/home/paperspace/code/aru_sil_core/src/scripts
TAG=glref
cd /home/paperspace/code/nerf_new
[ -d "$SUP" ] || { echo "REPL-FAIL: no supervision/trees_only in $BD"; exit 1; }
S1=$(ls -t $BD/splat_runs_STAGE1/stage1_bg00_${TAG}/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1)
[ -n "$S1" ] || { echo "REPL-FAIL: no stage1_bg00_${TAG} ckpt in $BD"; exit 1; }
echo "REPL-S1: $S1"
# 0. stage2_init: stage-1 (features-off) ckpt + zero 32-d high_features
if [ ! -f $BD/stage2_init_${TAG}/nerfstudio_models/$(basename $S1) ]; then
  pixi run python - "$S1" "$BD/stage2_init_${TAG}/nerfstudio_models" << 'PY' || exit 1
import sys, torch
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location='cpu')
means_key = [k for k in ck['pipeline'] if k.endswith('gauss_params.means')][0]
n = ck['pipeline'][means_key].shape[0]
hf_key = means_key.replace('means', 'high_features')
assert hf_key not in ck['pipeline'], 'stage1 ckpt already has features?'
ck['pipeline'][hf_key] = torch.zeros((n, 32))
dst.mkdir(parents=True, exist_ok=True)
torch.save(ck, dst / src.name)
print(f'REPL-INIT0: zero-feature stage2_init ({n} gaussians) -> {dst / src.name}')
PY
fi
# 1. bootstrap run (config only; 1 iteration)
if ! ls $BD/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml >/dev/null 2>&1; then
  echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_bootstrap_${TAG} \
      --load-dir $BD/stage2_init_${TAG}/nerfstudio_models \
      --pipeline.model.freeze-geometry True \
      --pipeline.model.high-loss-weight 1.0 \
      --pipeline.datamanager.semantic-dir $SUP \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.sky-loss-lambda 1.0 \
      --max-num-iterations 1 --steps-per-save 19998 \
      --vis tensorboard nerfstudio-data \
      --eval-mode interval --eval-interval 10 || echo "REPL-BOOTSTRAP-EXIT (config is all we need)"
fi
ls $BD/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml >/dev/null || { echo "REPL-FAIL: no bootstrap config"; exit 1; }
# 2. interaction census
W=$BD/splat_runs_FEATFIX/interaction_W_${TAG}.npz
[ -f $W ] || HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py \
  --run-glob "$BD/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml" \
  --supervision-dir $SUP --out-npz $W || { echo "REPL-FAIL: census"; exit 1; }
# 3. census-init ckpt
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py \
  --w-npz $W --embedder $EMB \
  --src-ckpt $BD/stage2_init_${TAG}/nerfstudio_models/$(basename $S1) \
  --dst-dir $BD/stage2_init_census_${TAG}/nerfstudio_models || { echo "REPL-FAIL: init build"; exit 1; }
printf '{"embedder": "%s", "hierarchy": "%s", "supervision": "%s", "stage1": "%s", "poses": "opengl (glref)"}\n' \
  "$EMB" "$HJ" "$SUP" "$S1" > $BD/splat_runs_FEATFIX/stage2_provenance_${TAG}.json
# 4. seed-only stage-2: stage the census-init ckpt as a run the render service can load
BOOT=$(ls -t $BD/splat_runs_FEATFIX/stage2_bootstrap_${TAG}/high/*/config.yml | head -1)
TS=$(basename $(dirname $BOOT))
RUN=$BD/splat_runs_FEATFIX/stage2_censusinit_${TAG}/high/$TS
mkdir -p $RUN/nerfstudio_models
cp $BD/stage2_init_census_${TAG}/nerfstudio_models/*.ckpt $RUN/nerfstudio_models/
sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_${TAG}|" $BOOT > $RUN/config.yml
DPT=$(dirname $BOOT)/dataparser_transforms.json
[ -f "$DPT" ] || DPT=$(ls -t $BD/splat_runs_STAGE1/stage1_bg00_${TAG}/high/*/dataparser_transforms.json 2>/dev/null | head -1)
if [ -n "$DPT" ] && [ -f "$DPT" ]; then cp "$DPT" $RUN/dataparser_transforms.json
else echo "REPL-WARN: no dataparser_transforms.json to stage — streamed render will refuse this block"; fi
ls $RUN/nerfstudio_models/*.ckpt >/dev/null || { echo "REPL-FAIL: seed not staged"; exit 1; }
echo "REPL-SEEDONLY: $N staged stage2_censusinit_${TAG}/high/$TS"
