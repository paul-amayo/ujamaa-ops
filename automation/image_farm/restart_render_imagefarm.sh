#!/bin/bash
# Restart only the :8004 checkpoint render service on an image_farm segment (scene server on :8001 untouched).
# RENDER_SKY defaults to black: these RGB splats were trained on a black background without foreground masks, so the
# citrus sky-blue composite only shows through partial-alpha pixels and outside the portrait training coverage.
#   usage: restart_render_imagefarm.sh <segment_dir>
SEG=$(realpath "${1:?segment dir}"); CFG=lio_arc_size15.0_ov0.10_kf20cm_dedup
pkill -f "render_service:ap[p] --host 127.0.0.1 --port 8004"; sleep 3
cd /home/paperspace/code/aru_sil_core && RENDER_BLOCKS_ROOT=$SEG/blocks_ns/$CFG RENDER_RUN_GLOB='splat_runs_high/*/high/*/config.yml' \
  RENDER_CKPT_POSES=opengl RENDER_SKY=${RENDER_SKY:-0,0,0} RENDER_VRAM_BUDGET=${RENDER_VRAM_BUDGET:-8} RENDER_PRELOAD=1 RENDER_MAX_BLOCKS=1 \
  setsid nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python -m uvicorn \
    src.interfaces.splat_viewer.render_service:app --host 127.0.0.1 --port 8004 >> /home/paperspace/logs/render_service_8004.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8004/healthz > /dev/null 2>&1 && break; sleep 3; done
echo "[restart] $(curl -s -m5 http://127.0.0.1:8004/healthz | cut -c1-200)"
