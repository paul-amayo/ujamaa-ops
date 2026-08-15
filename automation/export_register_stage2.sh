#!/bin/bash
# Export + register ONE two-stage block for the tassili viewer.
#   usage: export_register_stage2.sh <block_dir>
# Mirrors the unified pipeline's v16 export tail ([5b][6][7]) for a
# stage2_censusinit_fw2 checkpoint: ns-export -> crop -> .splat/.br ->
# features.npy -> back-attribution -> merge_block_splats (splats.json).
# Idempotent: every artifact is cache-checked. Distinct ply name keeps
# stage2 exports from colliding with v16 relics.
set -uo pipefail
BD=$(readlink -f "$1"); N=$(basename "$BD")
CFGDIR=$(dirname "$BD")
ROOT_DIR=$(dirname "$(dirname "$CFGDIR")")
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
LOG=/home/paperspace/logs/export_$(basename "$CFGDIR")_${N}.log
PLY=$BD/splats/splat_cropped_stage2_censusinit_fw2.ply
cd /home/paperspace/code/nerf_new

CFG=$(ls -t "$BD"/splat_runs_FEATFIX/stage2_censusinit_fw2/high/*/config.yml 2>/dev/null | head -1)
[ -n "$CFG" ] || { echo "EXPORT-SKIP $N: no stage2 config"; exit 0; }

if [ ! -f "$PLY" ]; then
  pixi run --manifest-path "$NS_PIXI" ns-export gaussian-splat \
    --load-config "$CFG" --output-dir "$BD/splats" --save-world-frame True \
    > "$LOG" 2>&1 || { echo "EXPORT-FAIL $N: ns-export"; exit 1; }
  python3 "$SRC/crop_splat_to_block.py" --block-dir "$BD" \
    --in "$BD/splats/splat.ply" --out "$PLY" >> "$LOG" 2>&1 \
    || { echo "EXPORT-FAIL $N: crop"; exit 1; }
fi
SPLAT=${PLY%.ply}.splat
if [ ! -f "$SPLAT" ] || [ "$PLY" -nt "$SPLAT" ]; then
  pixi run --manifest-path "$NS_PIXI" python "$SRC/ply_to_splat.py" \
    --in "$PLY" --out "$SPLAT" >> "$LOG" 2>&1 || echo "EXPORT-WARN $N: .splat failed"
fi
if [ -f "$SPLAT" ] && { [ ! -f "$SPLAT.br" ] || [ "$SPLAT" -nt "$SPLAT.br" ]; }; then
  pixi run --manifest-path "$NS_PIXI" python "$SRC/brotli_compress.py" "$SPLAT" \
    >> "$LOG" 2>&1 || echo "EXPORT-WARN $N: brotli failed"
fi

FEAT=${PLY%.ply}.features.npy
ATT=${PLY%.ply}.argmax_attribution.npy
if [ ! -f "$FEAT" ]; then
  pixi run --manifest-path "$NS_PIXI" python "$SRC/export_high_features.py" \
    --config "$CFG" --cropped-ply "$PLY" --block-dir "$BD" --out "$FEAT" \
    >> "$LOG" 2>&1 || echo "EXPORT-WARN $N: features failed"
fi
SG=$(ls -d "$ROOT_DIR"/scene_graph* 2>/dev/null | head -1)
if [ -f "$FEAT" ] && [ ! -f "$ATT" ] && [ -n "$SG" ]; then
  pixi run --manifest-path "$NS_PIXI" python "$SRC/compute_back_attribution.py" \
    --config "$CFG" --argmax-gate \
    --marker-monolithic "$(ls "$SG"/markers_v2*.monolithic 2>/dev/null | head -1)" \
    --semantic-monolithic "$ROOT_DIR/filtered_semantic_v2.monolithic" \
    --out "$ATT" >> "$LOG" 2>&1 || echo "EXPORT-WARN $N: attribution failed"
fi

# registration: stage_splat_manifests derives splats.json + index.json for
# the whole config from disk (idempotent — restaged on every block's export).
# merge_block_splats was the WRONG tool (it concatenates plys; the pipeline
# [7] call of it is a vestige from a CLI three renames ago).
if python3 "$SRC/stage_splat_manifests.py" \
    --data-root "$ROOT_DIR" --config "$(basename "$CFGDIR")" \
    --run stage2_censusinit_fw2 \
    --splat-rel splats/splat_cropped_stage2_censusinit_fw2.ply \
    >> "$LOG" 2>&1; then
  echo "EXPORT-DONE $N ($(du -m "$PLY" 2>/dev/null | cut -f1) MB ply$([ -f "$FEAT" ] && echo ' +features')$([ -f "$ATT" ] && echo ' +attr'), splats.json restaged)"
else
  echo "EXPORT-FAIL $N: splats.json staging"
  exit 1
fi
