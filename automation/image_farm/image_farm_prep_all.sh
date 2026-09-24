#!/bin/bash
# Prep-only sweep: frames + all-frame matching + segments + per-segment GPU SfM + track gate for every usable clip,
# so the training fleet finds sparse/0 ready. Skips clips whose segments already have capture_meta.json.
set -uo pipefail
F=/home/paperspace/data/image_farm
P=/home/paperspace/logs/image_farm_prep.py
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
CLIPS=${IF_CLIPS:-"IMG_7964 IMG_7971 IMG_7975 IMG_7990 IMG_7994 IMG_7995 IMG_7996 IMG_8001 IMG_7959 IMG_7961 IMG_7972 IMG_7974 IMG_7960 IMG_7970 IMG_7967 IMG_7963 IMG_7973 IMG_7984 IMG_7987 IMG_7993 IMG_7997 IMG_7998"}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] PREP-ALL $*"; }
for c in $CLIPS; do
  d=$F/$c; [ -d "$d" ] || { say "$c: missing"; continue; }
  if [ -f "$d/segments.json" ] && ls "$F/${c}"_s*/capture_meta.json > /dev/null 2>&1 && [ "$(ls -d "$F/${c}"_s*/ | wc -l)" -eq "$(python3 -c "import json; print(len(json.load(open('$d/segments.json'))['segments']))")" ] && ! ls "$F/${c}"_s*/ | grep -q "^$" ; then
    n_meta=$(ls "$F/${c}"_s*/capture_meta.json 2>/dev/null | wc -l); n_seg=$(python3 -c "import json; print(len(json.load(open('$d/segments.json'))['segments']))")
    [ "$n_meta" -eq "$n_seg" ] && { say "$c: prepped ($n_seg segments) — skip"; continue; }
  fi
  say "=== $c"; t0=$(date +%s)
  if "$PY" "$P" "$d" >> /home/paperspace/logs/prep_$c.log 2>&1; then
    say "=== $c done in $(( ($(date +%s)-t0)/60 )) min"; grep -E "segment\(s\)|kept|SFM FAILED" /home/paperspace/logs/prep_$c.log | tail -5
  else
    say "=== $c PREP FAILED after $(( ($(date +%s)-t0)/60 )) min (prep_$c.log)"
  fi
done
say "PREP-ALL COMPLETE"
