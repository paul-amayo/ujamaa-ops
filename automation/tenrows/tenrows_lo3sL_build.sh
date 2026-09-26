#!/bin/bash
# tenrows_lo3sL_build.sh — LiDAR-depth twin of the lo3s survey: same images, poses, chunks, LiDAR seed and scaffold, but
# the chunk training's depth prior is the LiDAR depth map (depths_lidar, metric inverse depth, sparse) instead of the
# DA-V2 monocular map. Needs tenrows_lidar_depth_maps.py to have run on the source project. Copies the project layout
# with hardlinks (images, scaffold, depth maps) and rewrites every chunk's depth_params.json for the LiDAR maps.
set -u
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows; S=$R/experimental/h3dgs_lo3s; X=$R/experimental/h3dgs_lo3sL
L=/home/paperspace/logs/tenrows_lo3sL_build.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -e $S/camera_calibration/rectified/depth_params_lidar.json ] || { say "no LiDAR depth maps on $S"; exit 1; }
mkdir -p $X/camera_calibration/rectified/images $X/camera_calibration/rectified/depths $X/output/scaffold/point_cloud/iteration_30000 $X/output/trained_chunks
for f in $S/camera_calibration/rectified/images/*.png; do ln -f "$f" $X/camera_calibration/rectified/images/$(basename "$f"); done
for f in $S/camera_calibration/rectified/depths_lidar/*.png; do ln -f "$f" $X/camera_calibration/rectified/depths/$(basename "$f"); done
cp -r $S/camera_calibration/aligned $S/camera_calibration/prior $S/camera_calibration/poses $X/camera_calibration/ 2>/dev/null; cp -r $S/camera_calibration/rectified/sparse $X/camera_calibration/rectified/ 2>/dev/null
cp -r $S/camera_calibration/chunks $X/camera_calibration/; cp $S/export_meta.json $X/
for f in point_cloud.ply pc_info.txt; do ln -f $S/output/scaffold/point_cloud/iteration_30000/$f $X/output/scaffold/point_cloud/iteration_30000/$f; done
python3 - $S $X <<'EOF'
import json, sys
from pathlib import Path
S, X = Path(sys.argv[1]), Path(sys.argv[2]); lp = json.load(open(S / "camera_calibration/rectified/depth_params_lidar.json"))
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary
for c in sorted((X / "camera_calibration/chunks").glob("*_*")):
    names = [im.name[:-4] for im in read_images_binary(str(c / "sparse/0/images.bin")).values()]
    dp = {n: lp[n] for n in names if n in lp}; json.dump(dp, open(c / "sparse/0/depth_params.json", "w")); print(f"[build] {c.name}: depth_params for {len(dp)}/{len(names)} images (LiDAR)")
EOF
say "built $X: $(ls $X/camera_calibration/rectified/images | wc -l) images, $(ls $X/camera_calibration/rectified/depths | wc -l) LiDAR depth maps, chunks $(ls $X/camera_calibration/chunks | tr '\n' ' '), scaffold hardlinked"
