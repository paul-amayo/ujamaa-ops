#!/bin/bash
# start_hier_service_lane.sh <lane project dir> [tau] — H3DGS hierarchy render backend (:8006, loopback) for a SINGLE-LANE
# project (ten_rows lane runs, 2026-09-27): one chunk 'lane', trained without a scaffold, so no skybox / no scaffold fill
# (SCAFFOLD is exported empty — CompactHierarchy skips it — and HIER_FILL=0); poses arrive in the LiDAR-odometry world
# the lane was trained in (export_meta world_rotation_to_zup = identity).
PROJ=${1:?lane h3dgs project dir}; TAU=${2:-3}
HIER=$PROJ/output/trained_chunks/lane/hierarchy.hier_opt; [ -e $HIER ] || { echo "[hier] no $HIER"; exit 1; }
for p in $(pgrep -f "^[^ ]*python [^ ]*hier_render_service\.py"); do kill -9 $p; done
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":8006 " || break; sleep 1; done
ss -ltn 2>/dev/null | grep -q ":8006 " && { echo "[hier] port 8006 still held: $(ss -ltnp | grep ':8006 ' | grep -oE 'pid=[0-9]+' | sort -u | tr '\n' ' ')"; exit 1; }
export HIER SCAFFOLD="" META=$PROJ/export_meta.json CHUNKS_DIR=$PROJ/camera_calibration/chunks MERGED_CHUNKS="lane" HIER_FILL=0 TAU PORT=8006
cd /home/paperspace/code/hierarchical-3d-gaussians
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup /home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer/hier_render_service.py > /home/paperspace/logs/hier_service_8006.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8006/healthz > /dev/null 2>&1 && break; sleep 3; done
echo "[hier] $(curl -s -m5 http://127.0.0.1:8006/healthz | cut -c1-300)"
