#!/bin/bash
# sidecar_chunks_stop.sh — stop the 05 chunk side-car chain loop and its current chunk's steps. Anchored patterns, run from this file.
R=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar_chunks
pkill -f "^bash -c for c in 0_1" && echo "chunk loop stopped" || echo "no chunk loop"
pkill -f "^bash [^ ]*sidecar_chunk\.sh 05_13D_Jackal" && echo "chunk driver stopped" || true
pkill -f "(sidecar_chunk_dataset|sidecar_dataparser|h3dgs_to_stage1)\.py .*(05_13D_Jackal|$R)" && echo "prep step stopped" || true
pkill -f "ns-train high --data $R/chunk_" && echo "bootstrap stopped" || true
pkill -f "(gaussian_interaction_census|build_census_init|containment_eval)\.py .*$R/chunk_" && echo "census/init/eval stopped" || true
sleep 4; left=$(pgrep -af "$R/chunk_" | grep -v chunks_stop); [ -z "$left" ] && echo "clean" || { echo "STILL RUNNING:"; echo "$left" | cut -c1-140; }
