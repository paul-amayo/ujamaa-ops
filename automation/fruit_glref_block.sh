#!/bin/bash
# Fruit densify + share-seed on the GL-fixed (glref) era of one block — the
# 2026-08-26 fruit recipe (fruit_chain.sh step 3c) ported to the models served
# since 09-21. The served glref seed stays untouched; the result is staged beside
# it as stage2_censusinit_glref_fruit, which a render glob of
# stage2_censusinit_glref*/high/*/config.yml picks (sorted last) on blocks that have it.
#
# Why (measured 2026-09-30, 05 b000 kf_000024, fruit 10001 = tree 5's 61 oranges):
# a fruit query on the glref seed lit the WHOLE parent canopy — the argmax-init
# failure of 08-26 — because the glref chain never ran the fruit densify pass.
#
#   usage: fruit_glref_block.sh <block_dir> <embedder_ckpt> [densify_iters]
set -u
BD=$(readlink -f "${1:?block dir}"); EMB=${2:?embedder ckpt}; ITERS=${3:-2000}
N=$(basename "$BD"); SUP=$BD/supervision/trees_fruit_v3
SCR=/home/paperspace/code/aru_sil_core/src/scripts
export PATH=/home/paperspace/.pixi/bin:$PATH
[ -f "$SUP/manifest.json" ] || { echo "FRUIT-GLREF-FAIL $N: no trees_fruit_v3 supervision"; exit 1; }
say() { echo "[$(date '+%H:%M:%S')] $N $*"; }

rm -rf "$BD/splat_runs_FEATFIX/fruit_densify_glref" "$BD/stage2_init_densify_glref"
say "densify (glref stage 1, $ITERS iters)"
DENSIFY_STAGE1=stage1_bg00_glref DENSIFY_TAG=_glref CENSUS_EMBEDDER="$EMB" \
  bash /home/paperspace/code/automation/densify_block.sh "$BD" "$SUP" "$ITERS" || exit 1
DRUN=$(ls -dt "$BD"/splat_runs_FEATFIX/fruit_densify_glref/high/*/ | head -1)
DCK=$(ls -t "$DRUN"/nerfstudio_models*/*.ckpt | head -1)

WD=$BD/splat_runs_FEATFIX/interaction_W_densified_glref.npz
say "census on the densified geometry"
(cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
    "$SCR/gaussian_interaction_census.py" --run-glob "$DRUN/config.yml" \
    --supervision-dir "$SUP" --out-npz "$WD") || { echo "FRUIT-GLREF-FAIL $N: census"; exit 1; }
python3 /home/paperspace/code/automation/fruit_diet_check.py "$WD" | grep DIET-MIN

say "share-seed ${FRUIT_SHARE_ASSIGN:-0.1}"
(cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
    "$SCR/build_census_init.py" --w-npz "$WD" --embedder "$EMB" --src-ckpt "$DCK" \
    --dst-dir "$BD/stage2_init_densify_glref/nerfstudio_models" \
    --fruit-share-assign "${FRUIT_SHARE_ASSIGN:-0.1}") || { echo "FRUIT-GLREF-FAIL $N: share-seed"; exit 1; }

TS=$(basename "$DRUN")
RUN=$BD/splat_runs_FEATFIX/stage2_censusinit_glref_fruit/high/$TS
rm -rf "$BD/splat_runs_FEATFIX/stage2_censusinit_glref_fruit"
mkdir -p "$RUN/nerfstudio_models"
sed 's|^experiment_name: .*$|experiment_name: stage2_censusinit_glref_fruit|' "$DRUN/config.yml" > "$RUN/config.yml"
cp "$DRUN/dataparser_transforms.json" "$RUN/" 2>/dev/null
cp "$BD"/stage2_init_densify_glref/nerfstudio_models/*.ckpt "$RUN/nerfstudio_models/"
say "FRUIT-GLREF-DONE staged $RUN"
