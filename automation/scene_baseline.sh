#!/bin/bash
# Run THE baseline for a scene, as declared in its scene.json manifest.
#   usage: scene_baseline.sh <block-dir> [experiment-name]
# The recipe (env, ns-train flags, supervision dir) comes from the manifest's
# "baseline" section — scripts read, they don't assume (TESTING.md §3).
# For citrus 04 that is treelod_bg00_v1: tree-only photometric L1, sky loss
# on, no depth supervision, canary armed.
set -e
BD=$(readlink -f "$1")
[ -d "$BD" ] || { echo "usage: scene_baseline.sh <block-dir>"; exit 1; }
# survey root = ancestor holding scene.json
ROOT=$BD
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/scene.json" ]; do ROOT=$(dirname "$ROOT"); done
[ -f "$ROOT/scene.json" ] || { echo "no scene.json above $BD — run scene_manifest.py build"; exit 1; }

eval "$(python3 - "$ROOT" << 'EOF'
import json, shlex, sys
sc = json.load(open(f"{sys.argv[1]}/scene.json"))
b = sc["baseline"]
print(f'NAME={shlex.quote(b["name"])}')
for k, v in b.get("env", {}).items():
    print(f'export {k}={shlex.quote(str(v))}')
if sc.get("supervision_dir"):
    print(f'export TREE_WEIGHT_DIR={shlex.quote(sys.argv[1] + "/" + sc["supervision_dir"])}')
print('FLAGS=' + shlex.quote(" ".join(f"--{k} {v}" for k, v in b["flags"].items())))
EOF
)"
EXP=${2:-baseline_$NAME}
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
echo "[scene_baseline] $ROOT -> recipe $NAME -> experiment $EXP"
set -x
cd /home/paperspace/code/nerf_new
T0=$(date +%s)
echo "n" | MAX_JOBS=4 pixi run ns-train high \
  --data "$BD" --output-dir "$BD/splat_runs_BASELINE" --experiment-name "$EXP" \
  --pipeline.datamanager.semantic-dir $EMPTY \
  $FLAGS \
  --vis tensorboard nerfstudio-data \
  --eval-mode interval --eval-interval 10 \
  || { echo "SCENE-BASELINE-FAIL $EXP"; exit 1; }
echo "SCENE-BASELINE-DONE $EXP in $(( ($(date +%s)-T0)/60 )) min"
