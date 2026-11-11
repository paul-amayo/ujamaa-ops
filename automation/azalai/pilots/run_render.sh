#!/bin/bash
# run_render.sh <rung dir> <tag> <cells> [extra args]
R=$1; TAG=$2; CELLS=$3; shift 3
cd /home/paperspace/code/nerf_new
setsid nohup /home/paperspace/.pixi/bin/pixi run python /mnt/shared/azalai_rr/bin/azalai_rr_render.py $R/prep.json $R/renders --cells $CELLS --tag $TAG "$@" > $R/render_$TAG.log 2>&1 &
echo $! > $R/render_$TAG.pid; echo started $(cat $R/render_$TAG.pid)
