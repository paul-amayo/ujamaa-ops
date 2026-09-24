#!/bin/bash
# image_farm_fleet.sh — phone clips in /home/paperspace/data/image_farm -> image-only splats, one segment at a time:
#   image_farm_prep.py (frames, all-frame matching, forward-walk SEGMENTS at the turn dips, per-segment GPU SfM + track
#   gate; segment survey dirs <clip>_s<k>) -> image_farm_recipe.sh per segment (advisory hierarchy gate, IF_MODE=rgb by
#   default: the HiGH feature chain needs a SAM3 concept that exists on these vegetable beds — 'tree' does not).
# Training flags come from the IMG_7999_s0 tests (2026-09-24): IF_ITERS=15000 IF_SCHED=1500 IF_STOP_SPLIT=10000
#   (eval 19.48 vs 15.77 at the recipe's 5000 it). If image_farm_prep_all.sh is running, a clip that it has not finished
#   yet is waited for (never two preps on one clip). A segment with a finished splat.ply is skipped.
# Skipped as unusable: IMG_7962 (0.4 s), IMG_7985 (1.2 s), IMG_7992 (2.9 s), IMG_7986 (5.8 s), IMG_8019 (11 s HEVC/HLG).
set -uo pipefail
F=/home/paperspace/data/image_farm
R=${IF_RECIPE:-/home/paperspace/logs/image_farm_recipe.sh}
P=/home/paperspace/logs/image_farm_prep.py
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
CFG=lio_arc_size15.0_ov0.10_kf20cm_dedup
export IF_ITERS=${IF_ITERS:-15000} IF_SCHED=${IF_SCHED:-1500} IF_STOP_SPLIT=${IF_STOP_SPLIT:-10000} IF_MODE=${IF_MODE:-rgb}
CLIPS=${IF_CLIPS:-"IMG_7999 IMG_7964 IMG_7971 IMG_7975 IMG_7990 IMG_7994 IMG_7995 IMG_7996 IMG_8001 IMG_7959 IMG_7961 IMG_7972 IMG_7974 IMG_7960 IMG_7970 IMG_7967 IMG_7963 IMG_7973 IMG_7984 IMG_7987 IMG_7993 IMG_7997 IMG_7998"}
PROMPT=${IF_PROMPT:-tree}; EPS=${IF_EPS:-0.3}; DET=${IF_DET:-30}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] FLEET $*"; }
prepped(){  # 0 when segments.json exists and every segment dir has capture_meta.json
  local d=$F/$1; [ -f "$d/segments.json" ] || return 1
  local n; n=$(python3 -c "import json; print(len(json.load(open('$d/segments.json'))['segments']))")
  [ "$(ls "$F/$1"_s*/capture_meta.json 2>/dev/null | wc -l)" -ge "$n" ]
}
say "recipe=$R mode=$IF_MODE iters=$IF_ITERS sched=$IF_SCHED stop_split=$IF_STOP_SPLIT prompt=$PROMPT"
for c in $CLIPS; do
  d=$F/$c
  [ -d "$d" ] || { say "$c: no such clip dir"; continue; }
  while ! prepped "$c" && pgrep -f "^bash /home/paperspace/logs/image_farm_prep_all.sh" > /dev/null; do sleep 60; done
  if ! prepped "$c"; then
    say "=== $c: prep (frames + matching + segments + per-segment GPU SfM)"
    if ! "$PY" "$P" "$d" >> /home/paperspace/logs/prep_$c.log 2>&1; then say "=== $c: PREP FAILED (prep_$c.log)"; continue; fi
    grep -E "segment\(s\)|kept|SFM FAILED" /home/paperspace/logs/prep_$c.log | tail -6
  fi
  for sd in $(ls -d "$F/${c}"_s*/ 2>/dev/null); do
    sd=${sd%/}; sn=$(basename "$sd")
    [ -f "$sd/capture_meta.json" ] || { say "$sn: no capture_meta (SfM failed) — skip"; continue; }
    OK=$(python3 -c "import json; m=json.load(open('$sd/capture_meta.json')); print(int(m['ok']))")
    SKY=$(python3 -c "import json; print(json.load(open('$sd/capture_meta.json'))['sky'])")
    [ "$OK" = "1" ] || { say "$sn: gate not ok (kept too few frames) — skip"; continue; }
    if [ -f "$sd/blocks_ns/$CFG/block_000/splats/splat.ply" ]; then say "$sn: splat.ply exists — skip"; continue; fi
    say "=== $sn: recipe start (prompt=$PROMPT eps=$EPS det=$DET sky=$SKY iters=$IF_ITERS)"
    t0=$(date +%s)
    if bash "$R" "$sd" --prompt "$PROMPT" --eps "$EPS" --det-dist "$DET" --sky "$SKY" --iters "$IF_ITERS" >> /home/paperspace/logs/fleet_$sn.log 2>&1; then
      say "=== $sn: DONE in $(( ($(date +%s)-t0)/60 )) min"
      grep -E "PSNR" /home/paperspace/logs/recipe_${sn}_train.log 2>/dev/null | tail -2
    else
      say "=== $sn: FAILED after $(( ($(date +%s)-t0)/60 )) min — $(grep -E 'GATE FAILED' /home/paperspace/logs/fleet_$sn.log | tail -1)"
    fi
  done
  df -h /home/paperspace/data | tail -1
done
say "FLEET COMPLETE"
