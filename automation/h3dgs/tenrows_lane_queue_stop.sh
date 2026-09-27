#!/bin/bash
# tenrows_lane_queue_stop.sh — Paul 2026-09-27 08:5x "stop the renders and prioritize side car for containment": stop the
# queued ablation arms (the v2/v3 queue drivers only — the arm currently in its run driver finishes) and every GPU render
# service (lane hierarchy backend :8006, image_farm render service :8004, ns-viewer). Anchored patterns, run from this file.
pkill -f "^bash [^ ]*tenrows_lane_ablation(_v[0-9]+)?\.sh( |$)" && echo "ablation queue drivers stopped (current arm finishes)" || echo "no queue driver"
/home/paperspace/logs/stop_hier_service.sh
pkill -f "uvicorn src.interfaces.splat_viewer.render_service:app" && echo "render service :8004 stopped" || echo "no :8004 render service"
pkill -f "^[^ ]*python[0-9.]* [^ ]*/ns-viewer --load-config" && echo "ns-viewer stopped" || echo "no ns-viewer"   # the pixi-run python process, not the wrapper
sleep 4; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do echo "$m  $(ps -o args= -p $p | cut -c1-80)"; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
