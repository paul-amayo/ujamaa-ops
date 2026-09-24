#!/bin/bash
# image_farm_fleet.sh — phone clips in /home/paperspace/data/image_farm -> image-only HiGH splats, one clip at a time:
#   image_farm_prep.py (frames + GPU SfM) -> prod_image_recipe.sh (--sky from capture_meta.json, gendia defaults otherwise).
# Order: validation clip, medium walks, long walks, short walks. Skipped as unusable: IMG_7962 (0.4 s), IMG_7985 (1.2 s),
# IMG_7992 (2.9 s), IMG_7986 (5.8 s), IMG_8019 (11 s, 720p HEVC/HLG). A clip with a finished splat.ply is skipped.
set -uo pipefail
F=/home/paperspace/data/image_farm
R=/home/paperspace/code/automation/prod_image_recipe.sh
P=/home/paperspace/logs/image_farm_prep.py
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
CFG=lio_arc_size15.0_ov0.10_kf20cm_dedup
CLIPS=${IF_CLIPS:-"IMG_7999 IMG_7964 IMG_7971 IMG_7975 IMG_7990 IMG_7994 IMG_7995 IMG_7996 IMG_8001 IMG_7959 IMG_7961 IMG_7972 IMG_7974 IMG_7960 IMG_7970 IMG_7967 IMG_7963 IMG_7973 IMG_7984 IMG_7987 IMG_7993 IMG_7997 IMG_7998"}
PROMPT=${IF_PROMPT:-tree}; EPS=${IF_EPS:-0.3}; DET=${IF_DET:-30}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] FLEET $*"; }
for c in $CLIPS; do
  d=$F/$c
  [ -d "$d" ] || { say "$c: no such clip dir"; continue; }
  if [ -f "$d/blocks_ns/$CFG/block_000/splats/splat.ply" ]; then say "$c: splat.ply exists — skip"; continue; fi
  say "=== $c: prep (frames + GPU SfM)"
  if ! "$PY" "$P" "$d" >> /home/paperspace/logs/prep_$c.log 2>&1; then say "=== $c: PREP FAILED (prep_$c.log)"; continue; fi
  tail -1 /home/paperspace/logs/prep_$c.log
  SKY=$(python3 -c "import json; print(json.load(open('$d/capture_meta.json'))['sky'])")
  say "=== $c: recipe start (prompt=$PROMPT eps=$EPS det=$DET sky=$SKY)"
  t0=$(date +%s)
  if bash "$R" "$d" --prompt "$PROMPT" --eps "$EPS" --det-dist "$DET" --sky "$SKY" >> /home/paperspace/logs/fleet_$c.log 2>&1; then
    say "=== $c: DONE in $(( ($(date +%s)-t0)/60 )) min"
    grep -E "PSNR" /home/paperspace/logs/recipe_${c}_train.log 2>/dev/null | tail -2
  else
    say "=== $c: FAILED after $(( ($(date +%s)-t0)/60 )) min — $(grep -E 'GATE FAILED' /home/paperspace/logs/fleet_$c.log | tail -1)"
  fi
  df -h /home/paperspace/data | tail -1
done
say "FLEET COMPLETE"
