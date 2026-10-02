#!/bin/bash
# Demo video v2 (2 Oct): the non-citrus walks re-shot through the live stream at 8 fps, stride 1, one pass each
# (Paul on v1: "clips too short and fast"). Citrus comes from the peer's reels, not from here.
#   klapmuts-dec25     lane walk, first 90 % of the records (the tunnel exit is blown out)
#   kendu-0514-plants  portrait, whole walk (61 records)
#   gwakungu-cabbage   portrait, whole walk (72 records) — registry now the global-id file (HUD names follow it)
# Frames -> /home/paperspace/data/demo_video_v2/<survey>/f_%05d.jpg + <survey>.mp4 (8 fps). Log: ~/logs/video_shots_v2.log
set -u
OUT=/home/paperspace/data/demo_video_v2; mkdir -p $OUT; L=/home/paperspace/logs/video_shots_v2.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python; CAP=/tmp/appcheck/capture_walk.py
stage(){ curl -s -X POST -H "content-type: application/json" -d "{\"survey\":\"$1\"}" localhost:8011/api/stage > /dev/null
  for i in $(seq 1 60); do sleep 5; r=$(curl -s localhost:8011/api/stage); echo "$r" | grep -q "\"state\":\"ready\"" && break; echo "$r" | grep -q -E "\"(failed|no_gpu)\"" && { say "stage $1 FAILED: $r"; return 1; }; done
  say "staged $1: $(echo $r | head -c 120)"; }
shot(){ sv=$1; shift; stage $sv || return 1; rm -rf $OUT/$sv; t0=$(date +%s)
  $PY $CAP $sv $OUT/$sv "$@" 2>&1 | grep -E "frames from|record|wrote" | tee -a $L; say "shot $sv in $(( $(date +%s)-t0 )) s"; }
# Klapmuts: 281 records at stride 2 on 1 Oct -> ~562 at stride 1; stop at 90 %.
N=505
shot klapmuts-dec25 --stride 1 --fps 8 --to $N
shot kendu-0514-plants --stride 1 --w 720 --h 1280 --fps 8
shot gwakungu-cabbage --stride 1 --w 720 --h 1280 --fps 8
curl -s -X POST localhost:8011/api/unstage > /dev/null; say "unstaged"
say "=== v2 shots done: $(ls $OUT/*.mp4 2>/dev/null | tr '\n' ' ')"; touch $OUT/DONE
