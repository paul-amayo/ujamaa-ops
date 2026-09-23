#!/bin/bash
# Point Tassili at a survey's H3DGS model: restart the :8001 scene server on that survey (trajectory = the
# hierarchy's own poses) and run the hierarchy backend scheduler on that project.
#   usage: tassili_serve_survey.sh <survey_root> <h3dgs_proj_dir> [keyframe for the browser check]
SURVEY=${1:?survey root}; PROJ=${2:?h3dgs project}; KF=${3:-}
L=/home/paperspace/logs/tassili_serve.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -e $PROJ/output/merged.hier ] || { say "no merged.hier under $PROJ — nothing to serve"; exit 1; }
say "=== serving $(basename $SURVEY) from $PROJ"
for p in $(pgrep -f "^/bin/bash /home/paperspace/logs/hier_backend_scheduler.sh"); do kill $p; done
/home/paperspace/logs/stop_hier_service.sh >> $L 2>&1
P8001=$(ss -ltnp 2>/dev/null | grep ":8001 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$P8001" ] && kill $P8001 && sleep 2
cd /home/paperspace/code/aru_sil_core
HIGH_SPLAT_CONFIG=lio_row100 HIGH_DATA_ROOT=$SURVEY HIGH_TRAJ_POSES=h3dgs setsid nohup /usr/bin/python3 -m uvicorn src.interfaces.splat_viewer.server:app --host 127.0.0.1 --port 8001 > /home/paperspace/logs/server_8001.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -sf -m3 "http://127.0.0.1:8001/scene/trajectory?stride=1000" > /dev/null 2>&1 && break; sleep 2; done
say "8001: $(curl -s -m10 'http://127.0.0.1:8001/scene/trajectory?stride=1000' | python3 -c 'import json,sys; d=json.load(sys.stdin); print("poses", d.get("poses"), "resolved", d.get("poses_resolved"), "/", d["count"], "of", d["total"])' 2>&1)"
H3DGS_PROJ=$PROJ setsid nohup /home/paperspace/logs/hier_backend_scheduler.sh > /home/paperspace/logs/hier_backend_scheduler.out 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -sf -m10 http://127.0.0.1:8006/healthz > /dev/null 2>&1 && break; sleep 5; done
say "8006: $(curl -s -m10 http://127.0.0.1:8006/healthz | grep -oE '"hier":"[^"]*"|"nodes":[0-9]+|"resident_gib":[0-9.]+' | tr '\n' ' ')"
if [ -n "$KF" ]; then
  cd /home/paperspace/logs && timeout 300 /home/paperspace/envs/hfeval_ft/bin/python hier_browser_check.py $KF 2>&1 | grep -aE "\[browser\]" | cut -c1-220 | tee -a $L
fi
say "SERVE DONE"
