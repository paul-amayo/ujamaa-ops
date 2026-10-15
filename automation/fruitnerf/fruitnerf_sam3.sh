#!/bin/bash
# fruitnerf_sam3.sh — SAM3 image mode 'apple' per tree: build_tree_instances.py only lists kf_<K>.png, so stage prod/scratch_sam3/kf_<K:06d>.png
# symlinks to the undistorted working set images/image_<K>.png (K = working index = lio image_idx).
ARU=/home/paperspace/code/aru_sil_core; SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python; LOGS=/home/paperspace/logs
for T in tree_01 tree_02 tree_03; do S=/home/paperspace/data/public/fruitnerf/high/$T; mkdir -p $S/prod/scratch_sam3
  for p in $S/images/image_*.png; do k=$(basename $p .png | cut -d_ -f2); ln -sf $p $S/prod/scratch_sam3/$(printf "kf_%06d.png" $k); done
  echo "[$(date '+%m-%d %H:%M:%S')] fruitnerf_$T: SAM3 'apple' on $(ls $S/prod/scratch_sam3 | wc -l) staged frames" | tee -a $LOGS/fruitnerf_$T.log
  $SAM3_PY $ARU/src/scripts/build_tree_instances.py --data-dir $S --prompt apple --min-mask-area 100 --no-skip-existing > $LOGS/fruitnerf_${T}_sam3.log 2>&1 || echo "SAM3 FAILED $T" | tee -a $LOGS/fruitnerf_$T.log
  tail -n 1 $LOGS/fruitnerf_${T}_sam3.log | tee -a $LOGS/fruitnerf_$T.log
done
