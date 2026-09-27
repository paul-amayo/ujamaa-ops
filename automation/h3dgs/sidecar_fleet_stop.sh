#!/bin/bash
# sidecar_fleet_stop.sh — stop the 05 block side-car fleet (Paul, 2026-09-27: chunk-level side-cars instead): the fleet
# driver first, then the per-block scripts and their python steps. Anchored patterns, run from this file.
R=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar
pkill -f "^bash [^ ]*sidecar_fleet_05\.sh" && echo "fleet driver stopped" || echo "no fleet driver"
pkill -f "^bash [^ ]*sidecar_r2\.sh 05_13D_Jackal" && echo "block r2 driver stopped" || true
pkill -f "^bash [^ ]*sidecar_block_glref\.sh 05_13D_Jackal" && echo "block chain driver stopped" || true
pkill -f "^bash [^ ]*sidecar_bg_reseed\.sh 05_13D_Jackal" && echo "reseed driver stopped" || true
pkill -f "^bash [^ ]*censusinit_block_glref\.sh $R/block_" && echo "census chain stopped" || true
pkill -f "h3dgs_to_stage1\.py --block .*05_13D_Jackal" && echo "converter stopped" || true
pkill -f "ns-train high --data $R/block_" && echo "bootstrap trainer stopped" || true
pkill -f "(gaussian_interaction_census|build_census_init|containment_eval)\.py .*$R/block_" && echo "census/init/eval stopped" || true
sleep 4; left=$(pgrep -af "$R/block_" | grep -v fleet_stop); [ -z "$left" ] && echo "clean" || { echo "STILL RUNNING:"; echo "$left" | cut -c1-140; }
