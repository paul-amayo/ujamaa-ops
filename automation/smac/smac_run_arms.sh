#!/usr/bin/env bash
# smac_run_arms.sh (30 Oct 2026) — the three SMaC arms on citrus, teach = 05_13D_Jackal, repeat = 04_13D_Jackal, scratch clone only.
#   segments : SegMatch segments of each keyframe -> MapClosures (her SMaC arm)          -> runs/<tag>_segments/estimated_trajectory_1.txt
#   raw      : the whole keyframe cloud -> MapClosures (the MapClosures baseline)          -> runs/<tag>_raw/estimated_trajectory_1.txt
#   descriptors : SegMatch's own descriptor matching of each repeat keyframe against the teach map as target cloud
#                 (truth/teach_map_05_z-0.2_v0.5.ply; the branch's seed target is a single scan) -> runs/<tag>_descriptors/matching_centroids.csv
# Usage: smac_run_arms.sh <tag> [n_teach] [n_repeat] [min_z] [density_res] [voxel] [arms]   (defaults: full splits 7528 / 7626, ground cut -0.2 m,
#        density map 0.2 m, keyframe voxel 0.5 m, arms "segments raw")
set -u
TAG=${1:?tag}; NT=${2:-7528}; NR=${3:-7626}; MINZ=${4:--0.2}; DENS=${5:-0.2}; VOX=${6:-0.5}; ARMS=${7:-segments raw}
B=/home/paperspace/code/aru_segmap_smac; D=/mnt/shared2/smac_citrus; R=$D/runs
export SMAC_TARGET_CLOUD=$D/05_13D_Jackal/laser/laser_1.ply SMAC_GROW_RADIUS=0.9 SMAC_MIN_CLUSTER=100
for ARM in $ARMS; do
  O=$R/${TAG}_$ARM; mkdir -p $O; cd $O
  APP_ARM=$ARM; export SMAC_TARGET_CLOUD=$D/05_13D_Jackal/laser/laser_1.ply
  if [ "$ARM" = descriptors ]; then APP_ARM=segments; export SMAC_TARGET_CLOUD=$D/truth/teach_map_05_z-0.2_v0.5.ply; fi
  echo "[$(date +%H:%M:%S)] arm=$ARM n_teach=$NT n_repeat=$NR min_z=$MINZ density_res=$DENS voxel=$VOX -> $O"
  $B/build/segmap_node --config_dir $B/config/kitti --python_dir $B/python --teach_dir $D/05_13D_Jackal --repeat_dir $D/04_13D_Jackal \
    --out_dir $O --n_teach $NT --n_repeat $NR --arm $APP_ARM --min_z $MINZ --density_res $DENS --voxel $VOX > run.log 2>&1; echo "[$(date +%H:%M:%S)] rc=$?"
  echo "  keyframes: $(ls $O | grep -c '^keyframe_')  closures: $(( $( [ -f $O/estimated_trajectory_1.txt ] && wc -l < $O/estimated_trajectory_1.txt || echo 1 ) - 1 ))  recognised rows: $( [ -f $O/matching_centroids.csv ] && wc -l < $O/matching_centroids.csv || echo 0 )  failed matchThread: $(grep -c 'matchThread failed' run.log)  crashes: $(grep -c 'stack trace' run.log)"
  rm -f $O/transformed_keyframe_*.ply   # the accumulated global cloud per closure, GBs on a long run
done
echo "[$(date +%H:%M:%S)] done"
