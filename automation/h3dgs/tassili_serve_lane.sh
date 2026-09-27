#!/bin/bash
# tassili_serve_lane.sh <lane dir> [h3dgs project name=h3dgs] [frame for the browser check] — point Tassili at a single-lane
# H3DGS checkpoint (Paul, 2026-09-27: "serve current checkpoint on tassili"): restart the :8001 scene server with the lane
# as its data root (trajectory = <lane>/lio_image_poses.json, the lane's TRAINED poses, replayed verbatim with
# HIGH_TRAJ_POSES=lio) and start the hierarchy backend on :8006 from the lane's hierarchy.hier_opt (no scaffold).
# Browser: http://localhost:8001/tassili/?stream_url=ws://127.0.0.1:8006/ws  (tunnel 8001 AND 8006).
LD=${1:?lane dir}; PROJ=$LD/${2:-h3dgs}; KF=${3:-}
L=/home/paperspace/logs/tassili_serve.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -e $PROJ/output/trained_chunks/lane/hierarchy.hier_opt ] || { say "no hierarchy.hier_opt under $PROJ — nothing to serve"; exit 1; }
[ -e $LD/lio_image_poses.json ] || { say "no $LD/lio_image_poses.json (trained-pose trajectory) — build it first"; exit 1; }
say "=== serving lane $(basename $LD) from $PROJ"
for p in $(pgrep -f "^/bin/bash /home/paperspace/logs/hier_backend_scheduler.sh"); do kill $p; done
/home/paperspace/logs/stop_hier_service.sh >> $L 2>&1
P8001=$(ss -ltnp 2>/dev/null | grep ":8001 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$P8001" ] && kill $P8001 && sleep 2
cd /home/paperspace/code/aru_sil_core
HIGH_SPLAT_CONFIG=lane HIGH_DATA_ROOT=$LD HIGH_TRAJ_POSES=lio setsid nohup /usr/bin/python3 -m uvicorn src.interfaces.splat_viewer.server:app --host 127.0.0.1 --port 8001 > /home/paperspace/logs/server_8001.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -sf -m3 "http://127.0.0.1:8001/scene/trajectory?stride=100" > /dev/null 2>&1 && break; sleep 2; done
say "8001: $(curl -s -m10 'http://127.0.0.1:8001/scene/trajectory?stride=100' | python3 -c 'import json,sys; d=json.load(sys.stdin); print("poses", d.get("poses"), "count", d["count"], "of", d["total"], "first", d["frames"][0]["image_name"], "last", d["frames"][-1]["image_name"])' 2>&1 | cut -c1-200)"
say "8001 endpoints: $(for e in scene/path scene/config scene/tree_to_block scene/blocks; do printf '%s=%s ' $e $(curl -s -o /dev/null -w '%{http_code}' -m10 http://127.0.0.1:8001/$e); done)"
bash /home/paperspace/logs/start_hier_service_lane.sh $PROJ 3 2>&1 | tail -1 | tee -a $L
if [ -n "$KF" ]; then
  cd /home/paperspace/logs && timeout 300 /home/paperspace/envs/hfeval_ft/bin/python hier_browser_check.py $KF 2>&1 | grep -aE "\[browser\]" | cut -c1-220 | tee -a $L
fi
say "SERVE DONE"
