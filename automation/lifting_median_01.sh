#!/bin/bash
# 01 global-id lifting, median depth anchor vs the legacy nearest-return anchor, same code, same params as the prod
# 01 run (global_ids.json 08-22: voxel 0.1, dilate 2, min_voxels 50, min_pts_per_voxel 1, split 0, depth-back 1.5,
# max-det-dist 10, ground 0.3, hygiene on). Masks from prod sam3_v2; results to experimental only (Paul 2026-10-03:
# "we have to use a better rule ... median points not closest points").
#   usage: lifting_median_01.sh <median|min>
set -uo pipefail
A=${1:?median|min}; S=/home/paperspace/data/citrus_all/01_13B_Jackal; O=$S/experimental/lifting_anchor_20261003/$A
mkdir -p $O; cd /home/paperspace/code/nerf_new
echo "[$(date '+%m-%d %H:%M:%S')] start anchor=$A -> $O"
nice -n 5 pixi run python /home/paperspace/code/aru_sil_core/src/scripts/cluster_tree_instances.py --data-dir $S \
    --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 --min-pts-per-voxel 1 --split-pitch-m 0 \
    --depth-back 1.5 --max-det-dist 10.0 --masks-name sam3_v2 --out-dir $O --depth-anchor $A --cache-points
echo "[$(date '+%m-%d %H:%M:%S')] done anchor=$A rc=$?"
