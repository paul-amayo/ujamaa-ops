#!/bin/bash
# Restart the demo render service (:8004) on the corrected 05 era. Kill pattern lives in this
# file, not in the caller's command line (pkill self-match trap).
S=/home/paperspace/data/citrus_all/05_13D_Jackal
pkill -f "render_service:ap[p] --host 127.0.0.1 --port 8004"; sleep 4
cd /home/paperspace/code/aru_sil_core && RENDER_BLOCKS_ROOT=$S/prod/tassili/blocks_ns/lio_row100 \
  RENDER_RUN_GLOB='splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml' RENDER_CKPT_POSES=opengl \
  HIGH_EMBEDDER_CKPT=$S/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth RENDER_VRAM_BUDGET=${RENDER_VRAM_BUDGET:-24} RENDER_PRELOAD=${RENDER_PRELOAD:-43} \
  setsid nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python -m uvicorn \
    src.interfaces.splat_viewer.render_service:app --host 127.0.0.1 --port 8004 >> /home/paperspace/logs/render_service_8004.log 2>&1 < /dev/null &
for i in $(seq 1 40); do curl -sf -m3 http://127.0.0.1:8004/healthz >/dev/null 2>&1 && break; sleep 3; done
echo "[restart] $(curl -s -m5 http://127.0.0.1:8004/healthz | cut -c1-120)"
