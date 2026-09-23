#!/bin/bash
# Start the H3DGS hierarchy render backend for 05_13D on :8006 (loopback). Usage: start_hier_service.sh [hier file] [merged chunk names...]
PROJ=${H3DGS_PROJ:-/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs}
HIER=${1:-$PROJ/output/merged_partial.hier}; shift
MERGED="${*:-0_0 0_2}"
for p in $(pgrep -f "^[^ ]*python [^ ]*hier_render_service\.py"); do kill -9 $p; done
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":8006 " || break; sleep 1; done   # wait for the port to free
ss -ltn 2>/dev/null | grep -q ":8006 " && { echo "[hier] port 8006 still held: $(ss -ltnp | grep ':8006 ' | grep -oE 'pid=[0-9]+' | sort -u | tr '\n' ' ')"; exit 1; }
export HIER SCAFFOLD=$PROJ/output/scaffold/point_cloud/iteration_30000 META=$PROJ/export_meta.json CHUNKS_DIR=$PROJ/camera_calibration/chunks MERGED_CHUNKS="$MERGED" TAU=${TAU:-3} PORT=8006
cd /home/paperspace/code/hierarchical-3d-gaussians
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup /home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer/hier_render_service.py > /home/paperspace/logs/hier_service_8006.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8006/healthz > /dev/null 2>&1 && break; sleep 3; done
echo "[hier] $(curl -s -m5 http://127.0.0.1:8006/healthz | cut -c1-300)"
