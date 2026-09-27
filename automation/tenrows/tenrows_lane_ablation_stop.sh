#!/bin/bash
# tenrows_lane_ablation_stop.sh — stop a lane ablation: the ablation driver first, then the lane run driver, SfM, trainer /
# hierarchy / post-opt / evaluator of any lane project. Anchored patterns (pgrep self-match trap).
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental
pkill -f "^bash [^ ]*tenrows_lane_ablation(_v[0-9]+)?\.sh( |$)" && echo "ablation driver stopped" || echo "no ablation driver"   # never matches this _stop script
pkill -f "^bash [^ ]*tenrows_lane_run.sh $R/lane[0-9]+( |$)" && echo "lane run driver stopped" || echo "no lane run driver"
pkill -f -- "--database_path $R/lane[0-9]+/colmap/database.db" && echo "SfM stopped" || true
pkill -f -- "--model_path $R/lane[0-9]+/h3dgs[^ ]*/output/trained_chunks/lane( |$)" && echo "trainer stopped" || echo "no trainer"
pkill -f "GaussianHierarchyCreator $R/lane[0-9]+/" && echo "hierarchy creator stopped" || true
pkill -f "h3dgs_eval_chunk.py $R/lane[0-9]+/" && echo "evaluator stopped" || true
sleep 4; left=$(pgrep -af "$R/lane[0-9]" | grep -v ablation_stop); [ -z "$left" ] && echo "clean: nothing left on the lanes" || { echo "STILL RUNNING:"; echo "$left" | cut -c1-160; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader
