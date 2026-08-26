#!/bin/bash
# CENSUS-INIT stage2 replication on a block — the recipe of record (2026-08-07).
#   usage: censusinit_block.sh <block_dir> <supervision_dir>
# Steps: (0) stage2_init = stage1_bg00 ckpt + zero 32-d high_features
#        (1) 1-iter bootstrap run (writes a config on this geometry)
#        (2) rasterizer-gradient interaction census -> W npz
#        (3) census-majority feature init ckpt
#        (4) real stage2 (fw2 recipe: fruit-weight 2.0, freeze geometry)
#        (5) verdict: aligned_gates + no-walk/walked pointing on top-fruit frame
# Embedder: v3vocab1k. Requires compiled supervision with the CURRENT vocab.
set -uo pipefail
BD=$(readlink -f "$1"); SUP=$(readlink -f "$2"); N=$(basename $BD)
# Env-overridable for non-04 surveys (unified pipeline passes both):
EMB=${CENSUS_EMBEDDER:-/home/paperspace/data/high/nerf/04_13D_v3vocab1k/ckpts/model_best.pth}
HJ=${CENSUS_HIERARCHY:-/home/paperspace/data/citrus_all/04_13D_Jackal/scene_graph_v4/marker_hierarchy_fruit5.json}
ARU=/home/paperspace/code/aru_sil_core/src/scripts
FIG=/home/paperspace/code/lab_notebook/figs
SCRATCH=/tmp/claude-1000/-home-paperspace-code/c47d8606-c5fe-4343-ae48-8faa25cdc994/scratchpad
cd /home/paperspace/code/nerf_new

S1=$(ls -t $BD/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt | head -1)
echo "REPL-S1: $S1"

# 0. stage2_init: add zero high_features to the stage1 (features-off) ckpt
if [ ! -f $BD/stage2_init/nerfstudio_models/$(basename $S1) ]; then
  pixi run python - "$S1" "$BD/stage2_init/nerfstudio_models" << 'PY' || exit 1
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
if ! ls $BD/splat_runs_FEATFIX/stage2_bootstrap/high/*/config.yml >/dev/null 2>&1; then
  echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
    pixi run ns-train high \
      --data $BD --output-dir $BD/splat_runs_FEATFIX --experiment-name stage2_bootstrap \
      --load-dir $BD/stage2_init/nerfstudio_models \
      --pipeline.model.freeze-geometry True \
      --pipeline.model.high-loss-weight 1.0 \
      --pipeline.datamanager.semantic-dir $SUP \
      --pipeline.model.rasterize-mode antialiased \
      --pipeline.model.sky-loss-lambda 1.0 \
      --max-num-iterations 1 --steps-per-save 19998 \
      --vis tensorboard nerfstudio-data \
      --eval-mode interval --eval-interval 10 || echo "REPL-BOOTSTRAP-EXIT (config is all we need)"
fi
ls $BD/splat_runs_FEATFIX/stage2_bootstrap/high/*/config.yml >/dev/null || { echo "REPL-FAIL: no bootstrap config"; exit 1; }

# 2. interaction census
W=$BD/splat_runs_FEATFIX/interaction_W.npz
[ -f $W ] || HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py \
  --run-glob "$BD/splat_runs_FEATFIX/stage2_bootstrap/high/*/config.yml" \
  --supervision-dir $SUP --out-npz $W || { echo "REPL-FAIL: census"; exit 1; }

# 3. census-init ckpt
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py \
  --w-npz $W --embedder $EMB \
  --src-ckpt $BD/stage2_init/nerfstudio_models/$(basename $S1) \
  --dst-dir $BD/stage2_init_census/nerfstudio_models || { echo "REPL-FAIL: init build"; exit 1; }

# provenance: the verdict must score with the SAME embedder+hierarchy the
# features were built from — a hand-kept map in the queue disagreed and
# scored apr with klapmuts_v1 / 04 with the v3vocab1k vocabulary (0.00).
# NB: written BEFORE the env block below — inserting it between those
# backslash-continued lines silently detached STAGE2_BD/SUP/EMBEDDER from
# the call (stage2 then trained nothing and exited 0; 8 night slots lost).
mkdir -p $BD/splat_runs_FEATFIX
printf '{"embedder": "%s", "hierarchy": "%s", "supervision": "%s"}\n' \
  "$EMB" "$HJ" "$SUP" > $BD/splat_runs_FEATFIX/stage2_provenance.json

# 4. stage2 — SEED-ONLY or the fw2 refine.
# CENSUS_SEED_ONLY=1 stops at the census-init seed: measured 2026-08-17 on
# 02 b000, 5000 iters of fw2 moved containment by <0.01 (0.964/0.777/0.706
# vs seed 0.964/0.774/0.707) once the target cache was fresh, so the refine
# costs ~45 GPU-min per block for nothing measurable. Seed-only doubles
# block throughput; the refine stays available for the A/B.
if [ "${CENSUS_SEED_ONLY:-0}" = "1" ]; then
  BOOT=$(ls -t $BD/splat_runs_FEATFIX/stage2_bootstrap/high/*/config.yml | head -1)
  TS=$(basename $(dirname $BOOT))
  RUN=$BD/splat_runs_FEATFIX/stage2_censusinit_seed/high/$TS
  mkdir -p $RUN/nerfstudio_models
  cp $BD/stage2_init_census/nerfstudio_models/*.ckpt $RUN/nerfstudio_models/
  sed 's|^experiment_name: .*$|experiment_name: stage2_censusinit_seed|' $BOOT > $RUN/config.yml
  # dataparser_transforms.json too: the seed is a staged CHECKPOINT COPY, not an
  # ns-train invocation, so nerfstudio never writes one here. Anything that maps
  # world coords into the run's dataparser frame then dies on a missing file —
  # the streamed render service (render_service.py) refuses every seed-only
  # block with FileNotFoundError (2026-08-22). Prefer the bootstrap's copy (same
  # stage, same --data $BD); fall back to stage1, which parses the same block.
  DPT=$(dirname $BOOT)/dataparser_transforms.json
  [ -f "$DPT" ] || DPT=$(ls -t $BD/splat_runs_STAGE1/*/high/*/dataparser_transforms.json 2>/dev/null | head -1)
  if [ -n "$DPT" ] && [ -f "$DPT" ]; then cp "$DPT" $RUN/dataparser_transforms.json
  else echo "REPL-WARN: no dataparser_transforms.json to stage — streamed render will refuse this block"; fi
  echo "REPL-SEEDONLY: staged census-init seed as stage2_censusinit_seed"
else
  STAGE2_BD=$BD STAGE2_SUP=$SUP STAGE2_EMBEDDER=$EMB \
  STAGE2_NAME=stage2_censusinit_fw2 STAGE2_FRUIT_W="${STAGE2_FRUIT_W:-2.0}" \
  STAGE2_INIT_DIR=$BD/stage2_init_census/nerfstudio_models \
    /home/paperspace/code/automation/stage2_fruitchild.sh
  ls $BD/splat_runs_FEATFIX/stage2_censusinit_fw2/high/*/nerfstudio_models*/*.ckpt \
    >/dev/null 2>&1 || { echo "REPL-FAIL: stage2 produced no ckpt"; exit 1; }
fi

# 5. verdict on the top-fruit frame
CFG=$(ls -t $BD/splat_runs_FEATFIX/stage2_censusinit_*/high/*/config.yml | head -1)
FR=$(pixi run python - "$SUP" << 'PY'
import sys
import numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16)
    n = int((a != 65535).sum())   # TREE-mode: ids>=200 is FRUIT-only and
    # returns no frame on fruitless surveys (02/05/klap) — REPL-FRAME: None
    if n > best[0]:
        best = (n, f.name)
print(best[1])
PY
)
echo "REPL-FRAME: $FR"
for MODE in "--no-walk" ""; do
  HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/fruit_pointing_map.py \
    --config $CFG --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP \
    --frame $FR $MODE \
    --out $FIG/repl_${N}_pointing$( [ -n "$MODE" ] && echo _nowalk ).png \
    2>/dev/null | grep -aE "CROSS-LEVEL|FP anatomy"
done
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/aligned_gates.py "$CFG" 2>/dev/null | tail -6
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/containment_eval.py \
  --config $CFG --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP \
  --frame $FR --kf-images $(dirname $(dirname $(dirname $BD)))/kf_images \
  --out $FIG/repl_${N}_containment.png 2>/dev/null | grep -aE "TREE|FRUIT|ROW|SAVED"
# ---- reclaim derivable intermediates -------------------------------------
# stage2_init      = stage1 ckpt + zero high_features   (rebuilt by step 0)
# stage2_init_census = that + the interaction census W  (rebuilt by steps 2-3)
# stage2_bootstrap ckpt = a 1-iteration run whose ONLY product is a config.yml
# All three are derivable from stage1 + interaction_W.npz in minutes, yet they
# cost ~2.9 G per block and were kept forever: measured 2026-08-22 on 05
# block_000 (967M x3 of a 6.2 G block). automation/hpc_block_chain.sh already
# drops the first two for scratch discipline; do it here so the fleet benefits.
# Escape hatch: CENSUS_KEEP_INTERMEDIATES=1 keeps them for debugging.
if [ "${CENSUS_KEEP_INTERMEDIATES:-0}" != "1" ]; then
  RECLAIM=$(du -xsck "$BD/stage2_init" "$BD/stage2_init_census" \
      "$BD"/splat_runs_FEATFIX/stage2_bootstrap/high/*/nerfstudio_models 2>/dev/null | tail -1 | cut -f1)
  rm -rf "$BD/stage2_init" "$BD/stage2_init_census"
  # Remove the WHOLE bootstrap run dir, not just its checkpoint. Step 1 gates on
  # stage2_bootstrap/high/*/config.yml existing, so deleting only
  # nerfstudio_models left the gate satisfied while the checkpoint it hands to
  # --load-dir was gone: the block succeeded once and then failed on EVERY
  # re-run with "No checkpoint directory found" (03 block_005, 2026-08-22).
  # Dropping the directory makes the gate fail, so a retry rebuilds it — and
  # step 0 already rebuilds stage2_init from the stage1 ckpt the same way.
  rm -rf "$BD"/splat_runs_FEATFIX/stage2_bootstrap
  echo "REPL-RECLAIM: dropped derivable intermediates ($(( ${RECLAIM:-0} / 1024 ))M) — CENSUS_KEEP_INTERMEDIATES=1 to keep"
fi

echo "REPL-DONE $N"
