#!/bin/bash
# SAM3 image-mode prompt probe on one clip's staged frames: how many masks does each concept produce?
# usage: image_farm_prompt_probe.sh <clip_dir> [prompt ...]   (outputs under <clip>/prod/bateleur/sam3_probe_<prompt>)
set -uo pipefail
CLIP=$(realpath "${1:?clip dir}"); shift
PROMPTS=${*:-"plant crop vegetable seedling"}
SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
S=/home/paperspace/code/aru_sil_core/src/scripts/build_tree_instances.py
NAME=$(basename "$CLIP")
for p in $PROMPTS; do
  "$SAM3_PY" "$S" --data-dir "$CLIP" --prompt "$p" --out-name "sam3_probe_$p" > /home/paperspace/logs/probe_${NAME}_$p.log 2>&1
  echo "[$(date '+%H:%M:%S')] $NAME prompt='$p': $(grep -oE "[0-9]+ masks" /home/paperspace/logs/probe_${NAME}_$p.log | awk '{s+=$1} END {print s+0}') masks over $(grep -oE "^\[clip [0-9]+\] [0-9]+ frames" /home/paperspace/logs/probe_${NAME}_$p.log | awk '{s+=$3} END {print s+0}') frames"
done
echo "probe done"
