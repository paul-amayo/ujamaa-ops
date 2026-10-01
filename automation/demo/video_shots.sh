#!/bin/bash
# Demo video, first draft (1 Oct): one shot per farm through the live stream, each staged in turn.
#   citrus-b-05  chunk 1_0 walk, tree 5 lit from record 40   (portrait no)
#   citrus-a-01  chunk 3_1 walk, tree 164 lit from record 60
#   klapmuts-dec25  lane walk (colour only)
#   kendu-0514-plants, gwakungu-cabbage  portrait walks (colour only)
# Frames -> /home/paperspace/data/demo_video/<survey>/f_%05d.jpg + <survey>.mp4 (24 fps). Log: ~/logs/video_shots.log
set -u
OUT=/home/paperspace/data/demo_video; mkdir -p $OUT; L=/home/paperspace/logs/video_shots.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python; CAP=/tmp/appcheck/capture_walk.py
stage(){ curl -s -X POST -H "content-type: application/json" -d "{\"survey\":\"$1\"}" localhost:8011/api/stage > /dev/null
  for i in $(seq 1 60); do sleep 5; r=$(curl -s localhost:8011/api/stage); echo "$r" | grep -q "\"state\":\"ready\"" && break; echo "$r" | grep -q -E "\"(failed|no_gpu)\"" && { say "stage $1 FAILED: $r"; return 1; }; done
  if echo "$r" | grep -q identity_ready; then for i in $(seq 1 30); do curl -s localhost:8025/healthz | grep -q "\"resident\":\[0" && break; sleep 4; done; fi
  say "staged $1: $(echo $r | head -c 120)"; }
shot(){ sv=$1; shift; stage $sv || return 1; rm -rf $OUT/$sv; t0=$(date +%s)
  $PY $CAP $sv $OUT/$sv "$@" 2>&1 | grep -E "frames from|record|wrote" | tee -a $L; say "shot $sv in $(( $(date +%s)-t0 )) s"; }
shot citrus-b-05 --stride 2 --light tree:5@40
shot citrus-a-01 --stride 2 --light tree:164@60
shot klapmuts-dec25 --stride 2
shot kendu-0514-plants --stride 1 --w 720 --h 1280
shot gwakungu-cabbage --stride 1 --w 720 --h 1280
say "=== all shots done: $(ls $OUT/*.mp4 2>/dev/null | tr '\n' ' ')"
