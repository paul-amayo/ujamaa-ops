#!/bin/bash
for p in $(pgrep -f "hier_render_service.py"); do kill -9 $p && echo "killed hierarchy backend pid $p"; done
sleep 3; n=$(pgrep -fc "hier_render_service.py"); echo "hierarchy backend processes left: $n"
