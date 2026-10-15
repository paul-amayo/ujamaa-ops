#!/bin/bash
# fruitnerf_registry_chain.sh — per tree (FruitNeRF poses, undistorted 1620x1080 working set from fruitnerf_poses.py): SAM3 image mode 'apple'
# (min mask 100 px) -> DA3 windowed depth -> fruit_registry_count.py vs the hand count.
ARU=/home/paperspace/code/aru_sil_core; IPL=$ARU/src/scripts/image_pipeline; SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python; NS=/home/paperspace/code/nerf_new; LOGS=/home/paperspace/logs
declare -A GT=([tree_01]=179 [tree_02]=113 [tree_03]=291)
for T in tree_01 tree_02 tree_03; do S=/home/paperspace/data/public/fruitnerf/high/$T; L=$LOGS/fruitnerf_$T.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] fruitnerf_$T: $*" | tee -a $L; }
  if [ -z "$(ls $S/prod/bateleur/sam3_v2/clip_*/frame_entries.json $S/sam3_v2/clip_*/frame_entries.json 2>/dev/null | head -1)" ]; then
    say "SAM3 image mode 'apple' on the undistorted working set"; $SAM3_PY $ARU/src/scripts/build_tree_instances.py --data-dir $S --images-dir $S/images --prompt apple --min-mask-area 100 > $LOGS/fruitnerf_${T}_sam3.log 2>&1 || { say "SAM3 FAILED"; continue; }; fi
  if [ -z "$(ls $S/depth_png 2>/dev/null | head -1)" ]; then
    say "DA3 windowed depth"; ( cd /home/paperspace/code/sam3 && $SAM3_PY $IPL/da3_windows_fuse.py --survey $S --write-depth --depth-size 1620x1080 --dmax 30 ) > $LOGS/fruitnerf_${T}_da3.log 2>&1 || { say "DA3 FAILED"; continue; }; fi
  say "registry"; ( cd $NS && pixi run python /home/paperspace/code/automation/fruitnerf/fruit_registry_count.py $S --gt ${GT[$T]} 2>&1 ) | grep -aE "^\[registry\]|Error|Traceback" | tee -a $L
done
echo "[$(date '+%m-%d %H:%M:%S')] === fruitnerf registry chain done"
