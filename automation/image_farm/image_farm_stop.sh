#!/bin/bash
# Stop the image_farm fleet cleanly: driver first (so it cannot start the next clip), then the running prep / recipe
# and their children. Anchored patterns; run from this file, never inline (pgrep self-match trap).
for pat in "^bash /home/paperspace/logs/image_farm_fleet.sh" "^bash /home/paperspace/logs/image_farm_launch.sh" \
           "^bash /home/paperspace/code/automation/prod_image_recipe.sh" "^bash /home/paperspace/logs/image_farm_recipe.sh" \
           "^/home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/logs/image_farm_prep.py" \
           "^/home/paperspace/code/glomap/build_gpu/" \
           "^/home/paperspace/code/sam3/.pixi/envs/default/bin/python /home/paperspace/code/aru_sil_core/src/scripts/(build_tree_instances|build_sky_masks|build_fg_masks).py" \
           "^/home/paperspace/code/sam3/.pixi/envs/default/bin/python /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/da3_windows_fuse.py" \
           "ns-train high .*image_farm" "ns-export gaussian-splat .*image_farm"; do
  n=$(pgrep -f "$pat" | wc -l); [ "$n" -gt 0 ] && { echo "killing $n: $pat"; pkill -f "$pat"; }
done
sleep 2; pgrep -af "image_farm|prod_image_recipe" | grep -v -E "pgrep|image_farm_stop" | cut -c1-120
echo "stop done"
