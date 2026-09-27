#!/bin/bash
# tenrows_lane_fleet.sh [lanes...] — scale the proven lane-2 recipe to the other ten_rows lanes (Paul, 2026-09-27: "go for
# it"), one lane at a time on the single GPU: extract the lane's full 15 Hz left stream (window from lane_windows.json,
# skipped when <lane>/stamps.json exists), then tenrows_lane_run.sh (SfM, warp onto the LiDAR odometry, LiDAR init,
# train_single 60k / densify to 45k / grad 0.0075 under an 8 M gaussian budget, hierarchy, train_post 30k, evals + renders).
# Disk guard: a lane needs ~10 GB; skipped when the data volume has < 25 GB free. Default lanes: all but lane 2 (done).
set -u
LANES=${@:-1 3 4 5 6 7 8 9 10}
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental; WJ=$R/lane_windows.json; L=/home/paperspace/logs/tenrows_lane_fleet.log
XPY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10; XPP=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== fleet start: lanes $LANES"
for n in $LANES; do
  LD=$R/lane$n; free=$(df --output=avail -BG /home/paperspace/data | tail -1 | tr -dc 0-9)
  [ "$free" -ge 25 ] || { say "lane $n SKIPPED: only ${free}G free on the data volume"; continue; }
  if [ ! -e $LD/stamps.json ]; then
    read t0 t1 < <(python3 -c "import json;w=json.load(open('$WJ'))['$n'];print(w['t0_ms'],w['t1_ms'])")
    PYTHONPATH=$XPP env -u LD_LIBRARY_PATH -u LD_PRELOAD $XPY /home/paperspace/logs/tenrows_lane_extract.py $t0 $t1 $LD > /home/paperspace/logs/tenrows_lane${n}_extract.log 2>&1 || { say "lane $n EXTRACT FAILED"; continue; }
    say "lane $n: $(tail -1 /home/paperspace/logs/tenrows_lane${n}_extract.log | cut -c1-140)"
  fi
  t0s=$(date +%s)
  bash /home/paperspace/logs/tenrows_lane_run.sh $LD 60000 45000 0.0075 8000000 > /home/paperspace/logs/tenrows_lane${n}_run.out 2>&1
  say "lane $n finished in $(( ($(date +%s)-t0s)/60 )) min: $(grep -aE 'SfM in|lane-warp|HELD-OUT|TRAINING|REFUSED|FAILED|NO |DONE' /home/paperspace/logs/tenrows_lane_lane$n.log | tail -6 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-170 | tr '\n' '|')"
done
say "=== fleet DONE"
