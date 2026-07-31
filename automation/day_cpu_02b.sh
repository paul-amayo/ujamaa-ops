#!/usr/bin/env bash
# 02_13B ingest chain (CPU, safe alongside GPU training), following the proven
# run_03_pipeline.sh recipe: combined.bag -> monolithics -> LIO (micro_imu!) ->
# lio_image_poses.json -> kf20cm filter. Idempotent per stage.
set -u
ROOT=/home/paperspace/data/citrus_all/02_13B_Jackal
BIN=/home/paperspace/code/aru_sil_core/build/bin
PYTHON_PATH=/home/paperspace/code/aru_sil_core/src/utilities/ros/python/
SRC=/home/paperspace/code/aru_sil_core/src/scripts
LOG=$ROOT/02_ingest.log

echo "=== 02 ingest start $(date -u) ===" | tee -a "$LOG"

if [ ! -f "$ROOT/laser.monolithic" ] || [ ! -f "$ROOT/micro_imu.monolithic" ]; then
  echo "[mono] zed_bag_to_monolithics ($(date -u +%H:%M))" | tee -a "$LOG"
  "$BIN/zed_bag_to_monolithics" \
      -BAG_FILE "$ROOT/combined.bag" \
      -MONO_PATH "$ROOT/" \
      -PYTHON_PATH "$PYTHON_PATH" \
      -logtostderr >> "$LOG" 2>&1
  [ -f "$ROOT/laser.monolithic" ] && [ -f "$ROOT/micro_imu.monolithic" ] || {
      echo "[mono] FAILED — missing laser/micro_imu monolithic" | tee -a "$LOG"; exit 1; }
else
  echo "[mono] monolithics already present, skip" | tee -a "$LOG"
fi

if [ ! -f "$ROOT/transform_lio.monolithic" ]; then
  echo "[lio] fast_lio_from_mono (micro_imu — NOT imu) ($(date -u +%H:%M))" | tee -a "$LOG"
  "$BIN/fast_lio_from_mono" \
      -LASER_MONO "$ROOT/laser.monolithic" \
      -IMU_MONO   "$ROOT/micro_imu.monolithic" \
      -TRANSFORM_MONO "$ROOT/transform_lio.monolithic" \
      -logtostderr >> "$LOG" 2>&1
  [ -f "$ROOT/transform_lio.monolithic" ] || { echo "[lio] FAILED" | tee -a "$LOG"; exit 1; }
else
  echo "[lio] transform_lio present, skip" | tee -a "$LOG"
fi

if [ ! -f "$ROOT/lio_image_poses.json" ]; then
  echo "[dump] dump_lio_image_poses ($(date -u +%H:%M))" | tee -a "$LOG"
  "$BIN/dump_lio_image_poses" \
      -IMAGE_MONO "$ROOT/image_left.monolithic" \
      -TRANSFORM_MONO "$ROOT/transform_lio.monolithic" \
      -OUT_JSON "$ROOT/lio_image_poses.json" \
      -logtostderr >> "$LOG" 2>&1
  [ -f "$ROOT/lio_image_poses.json" ] || { echo "[dump] FAILED" | tee -a "$LOG"; exit 1; }
fi

if [ ! -f "$ROOT/lio_image_poses_kf20cm.json" ]; then
  echo "[kf] keyframe_filter_lio ($(date -u +%H:%M))" | tee -a "$LOG"
  python3 "$SRC/keyframe_filter_lio.py" \
      --in "$ROOT/lio_image_poses.json" \
      --out "$ROOT/lio_image_poses_kf20cm.json" \
      --min-step 0.20 --max-step 5.0 >> "$LOG" 2>&1 || { echo "[kf] FAILED" | tee -a "$LOG"; exit 1; }
fi

echo "=== 02 INGEST COMPLETE $(date -u) ===" | tee -a "$LOG"
echo "Next: GPU stages (SAM3/hierarchy) at GPU handover, then registry + G3 association."
