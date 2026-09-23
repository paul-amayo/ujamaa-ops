#!/bin/bash
# Stop a running H3DGS survey instance (its bash loop and the worker it is currently running) and the citrus queue.
SV=${1:?survey root}
for p in $(pgrep -f "^/bin/bash /home/paperspace/logs/h3dgs_queue_(citrus|all).sh"); do kill $p && echo "stopped queue $p"; done
for p in $(pgrep -f "^/bin/bash /home/paperspace/logs/h3dgs_survey.sh $SV "); do kill $p && echo "stopped survey loop $p"; done
sleep 1
for p in $(pgrep -f "^[^ ]*python( -u)? (/home/paperspace/logs/h3dgs_depth_05.py|/home/paperspace/logs/h3dgs_export.py|train_single.py|train_post.py|train_coarse.py)|^[^ ]*/colmap "); do
  # only workers of THIS survey: its project path in the command line, or H3DGS_PROJ in the environment
  if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$SV/" || tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^H3DGS_PROJ=$SV/"; then
    kill $p && echo "stopped worker $p ($(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-60))"; fi; done
