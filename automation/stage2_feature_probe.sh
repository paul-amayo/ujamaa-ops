#!/bin/bash
# STAGE2 FEATURE PROBE — where does the field lose tree identity?
# Paul 2026-08-17: "geometry is frozen, only the high features change, so how
# can this be destroyed. do only 1k iterations and then test features on
# 1, 100, and 1000."
#
# Runs the EXACT stage2 recipe from the census-init seed for 1000 iterations,
# saving every 100, then scores containment at steps 1 / 100 / 1000 on one
# frame, and prints the HIGH loss curve only.
#   usage: stage2_feature_probe.sh <block_dir> <sup_dir> <embedder> <hierarchy> <frame>
set -uo pipefail
BD=$(readlink -f "$1"); SUP=$(readlink -f "$2"); EMB=$3; HJ=$4; FRAME=$5
ARU=/home/paperspace/code/aru_sil_core/src/scripts
FIG=/home/paperspace/code/lab_notebook/figs
LOG=/home/paperspace/logs/stage2_probe_$(basename "$BD").log
KF=$(python3 -c "p='$BD'.split('/blocks_ns/')[0]; print(p+'/kf_images')")
cd /home/paperspace/code/nerf_new

# --- 1-iteration run (step 1 checkpoint) ----------------------------------
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
  pixi run ns-train high \
    --data "$BD" --output-dir "$BD/splat_runs_PROBE" --experiment-name probe_s1 \
    --load-dir "$BD/stage2_init_census/nerfstudio_models" \
    --pipeline.model.freeze-geometry True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight 2.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.datamanager.semantic-dir "$SUP" \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --max-num-iterations 15001 --steps-per-save 1 \
    --save-only-latest-checkpoint False \
    --vis tensorboard --logging.local-writer.enable False \
    nerfstudio-data --eval-mode interval --eval-interval 10 \
    > "$LOG.s1" 2>&1 || echo "PROBE-S1-EXIT"

# --- 1000-iteration run, saving every 100 --------------------------------
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
  pixi run ns-train high \
    --data "$BD" --output-dir "$BD/splat_runs_PROBE" --experiment-name probe_s1000 \
    --load-dir "$BD/stage2_init_census/nerfstudio_models" \
    --pipeline.model.freeze-geometry True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight 2.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.datamanager.semantic-dir "$SUP" \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --max-num-iterations 20001 --steps-per-save 100 \
    --save-only-latest-checkpoint False \
    --vis tensorboard --logging.local-writer.enable False \
    nerfstudio-data --eval-mode interval --eval-interval 10 \
    > "$LOG" 2>&1 || echo "PROBE-S1000-EXIT"

echo "=== HIGH loss only ==="
grep -aoE "High Loss[^ ]*[ ]+[0-9.e-]+|high_loss[^ ]*[ ]+[0-9.e-]+" "$LOG" \
  | tail -20 || echo "(no High Loss lines — check $LOG)"

# --- score containment at steps 1 / 100 / 1000 ---------------------------
score () {  # $1 = ckpt path, $2 = tag
  local CK=$1 TAG=$2
  local SRC DST TS
  SRC=$(ls -t "$BD"/splat_runs_PROBE/probe_s1000/high/*/config.yml 2>/dev/null | head -1)
  [ -n "$SRC" ] || SRC=$(ls -t "$BD"/splat_runs_PROBE/probe_s1/high/*/config.yml | head -1)
  TS=$(basename "$(dirname "$SRC")")
  DST=$BD/splat_runs_PROBE/eval_$TAG/high/$TS
  mkdir -p "$DST/nerfstudio_models"
  cp "$CK" "$DST/nerfstudio_models/"
  sed "s|^experiment_name: .*$|experiment_name: eval_$TAG|" "$SRC" > "$DST/config.yml"
  echo "--- containment @ $TAG ($(basename "$CK")) ---"
  HIGH_EMBEDDER_CKPT=$EMB pixi run python "$ARU/containment_eval.py" \
    --config "$DST/config.yml" --hyper-ckpt "$EMB" --hierarchy-json "$HJ" \
    --supervision-dir "$SUP" --frame "$FRAME" --kf-images "$KF" \
    --out "$FIG/probe_${TAG}_$(basename "$BD").png" 2>/dev/null \
    | grep -aE "TREE |ROW " || echo "(no verdict lines)"
}

S1=$(ls "$BD"/splat_runs_PROBE/probe_s1/high/*/nerfstudio_models/*000015001.ckpt 2>/dev/null | head -1)
[ -n "$S1" ] && score "$S1" step1
for W in 15100 16000 20000; do
  CK=$(ls "$BD"/splat_runs_PROBE/probe_s1000/high/*/nerfstudio_models/*$(printf '%09d' $W).ckpt 2>/dev/null | head -1)
  [ -n "$CK" ] && score "$CK" step$((W - 15000))
done
echo "PROBE-DONE"
