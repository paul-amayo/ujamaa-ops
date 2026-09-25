#!/bin/bash
# Point Tassili at one image_farm segment (nerfstudio checkpoint, RGB HiGH model): restart the :8001 scene server on the
# segment (trajectory = the ingest's OpenCV poses, which ARE the trained poses for image-only surveys) and start the
# checkpoint render service on :8004 (OpenGL-trained checkpoint -> RENDER_CKPT_POSES=opengl). Stops the H3DGS hierarchy
# backend (:8006) first so no stale backend answers for a different survey. Kill patterns live here (pkill self-match).
#   usage: tassili_serve_imagefarm.sh <segment_dir e.g. /home/paperspace/data/image_farm/IMG_7994_s0>
SEG=$(realpath "${1:?segment dir}"); CFG=lio_arc_size15.0_ov0.10_kf20cm_dedup
L=/home/paperspace/logs/tassili_serve.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -f "$SEG/blocks_ns/$CFG/block_000/splats/splat.ply" ] || { say "no finished splat under $SEG"; exit 1; }
say "=== serving image_farm segment $(basename $SEG)"
for p in $(pgrep -f "^/bin/bash /home/paperspace/logs/hier_backend_scheduler.sh"); do kill $p; done
/home/paperspace/logs/stop_hier_service.sh >> $L 2>&1
pkill -f "render_service:ap[p] --host 127.0.0.1 --port 8004"; sleep 2
P8001=$(ss -ltnp 2>/dev/null | grep ":8001 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$P8001" ] && kill $P8001 && sleep 2
cd /home/paperspace/code/aru_sil_core
HIGH_SPLAT_CONFIG=$CFG HIGH_DATA_ROOT=$SEG HIGH_TRAJ_POSES=lio setsid nohup /usr/bin/python3 -m uvicorn src.interfaces.splat_viewer.server:app --host 127.0.0.1 --port 8001 > /home/paperspace/logs/server_8001.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -sf -m3 "http://127.0.0.1:8001/scene/trajectory?stride=1000" > /dev/null 2>&1 && break; sleep 2; done
say "8001: $(curl -s -m10 'http://127.0.0.1:8001/scene/trajectory?stride=1000' | python3 -c 'import json,sys; d=json.load(sys.stdin); print("poses", d.get("poses"), "resolved", d.get("poses_resolved"), "/", d["count"], "of", d["total"], "owner", d["frames"][0].get("block"))' 2>&1)"
say "8001 blocks: $(curl -s -m10 http://127.0.0.1:8001/scene/blocks | cut -c1-200)"
RENDER_BLOCKS_ROOT=$SEG/blocks_ns/$CFG RENDER_RUN_GLOB='splat_runs_high/*/high/*/config.yml' RENDER_CKPT_POSES=opengl \
  RENDER_VRAM_BUDGET=${RENDER_VRAM_BUDGET:-8} RENDER_PRELOAD=1 RENDER_MAX_BLOCKS=1 \
  setsid nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python -m uvicorn \
    src.interfaces.splat_viewer.render_service:app --host 127.0.0.1 --port 8004 >> /home/paperspace/logs/render_service_8004.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -m3 http://127.0.0.1:8004/healthz > /dev/null 2>&1 && break; sleep 3; done
say "8004: $(curl -s -m5 http://127.0.0.1:8004/healthz | cut -c1-300)"
say "SERVE DONE — open http://localhost:8001/tassili/?stream_url=ws://localhost:8004/ws (tunnel 8001 and 8004)"
