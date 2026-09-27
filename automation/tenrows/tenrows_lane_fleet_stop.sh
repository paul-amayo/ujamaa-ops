#!/bin/bash
# tenrows_lane_fleet_stop.sh — stop the lane fleet: the fleet driver first, then the current lane's run driver, its SfM
# binaries (colmap / glomap on the lane's database) and its trainer / post-opt / evaluator. Anchored patterns (pgrep self-match trap).
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental
pkill -f "^bash [^ ]*tenrows_lane_fleet.sh" && echo "fleet driver stopped" || echo "no fleet driver"
pkill -f "^bash [^ ]*tenrows_lane_run.sh $R/lane[0-9]+( |$)" && echo "lane run driver stopped" || echo "no lane run driver"
pkill -f -- "--database_path $R/lane[0-9]+/colmap/database.db" && echo "SfM stopped" || echo "no SfM"
pkill -f -- "--model_path $R/lane[0-9]+/h3dgs[^ ]*/output/trained_chunks/lane( |$)" && echo "trainer stopped" || echo "no trainer"
pkill -f "GaussianHierarchyCreator $R/lane[0-9]+/" && echo "hierarchy creator stopped" || true
pkill -f "h3dgs_eval_chunk.py $R/lane[0-9]+/" && echo "evaluator stopped" || true
sleep 4; left=$(pgrep -af "$R/lane[0-9]" | grep -v fleet_stop); [ -z "$left" ] && echo "clean: nothing left on the lanes" || { echo "STILL RUNNING:"; echo "$left" | cut -c1-160; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader
