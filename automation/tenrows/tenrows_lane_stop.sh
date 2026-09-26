#!/bin/bash
# tenrows_lane_stop.sh <lane dir> — stop a ten_rows lane run: the driver first (so it cannot start the next stage), then
# its trainer / post-opt / evaluator. Anchored patterns, run from this file (pgrep/pkill self-match trap).
LD=${1:?lane dir}
pkill -f "^bash [^ ]*tenrows_lane_run.sh $LD( |$)" && echo "driver stopped" || echo "no driver"
pkill -f -- "--model_path $LD/h3dgs/output/trained_chunks/lane( |$)" && echo "trainer stopped" || echo "no trainer"
pkill -f "h3dgs_eval_chunk.py $LD/h3dgs( |$)" && echo "evaluator stopped" || true
sleep 4
left=$(pgrep -af "$LD" | grep -v tenrows_lane_stop); [ -z "$left" ] && echo "clean: nothing left on $LD" || { echo "STILL RUNNING:"; echo "$left" | cut -c1-160; }
