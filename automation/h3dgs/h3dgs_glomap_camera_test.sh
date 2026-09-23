#!/bin/bash
# GLOMAP from scratch on ONE block's keyframes (poses free, no odometry involved) under several FIXED camera models,
# reporting each reconstruction's registered images, points and mean reprojection error — "does the camera reduce
# GLOMAP's BA error". A last run lets GLOMAP optimise the intrinsics from Paul's start.
#   usage: h3dgs_glomap_camera_test.sh <survey_root> <block_NNN> <undistorted images dir or ->
S=${1:?survey}; B=${2:?block}; UD=${3:--}
W=$S/experimental/glomap_camera_test_$B; mkdir -p $W; L=/home/paperspace/logs/h3dgs_glomap_camera_test.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
export LD_LIBRARY_PATH=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
# keyframe list of the block (original frames) + the undistorted twins
$PY - $S $B $W $UD << 'EOF'
import json, os, sys
from pathlib import Path
S, B, W, UD = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
t = json.load(open(S / "prod/tassili/blocks_ns/lio_row100" / B / "transforms.json"))
for kind, src in [("orig", None), ("undist", Path(UD) if UD != "-" else None)]:
    if kind == "undist" and src is None: continue
    d = W / f"images_{kind}"; d.mkdir(exist_ok=True); n = 0
    for f in t["frames"]:
        p = Path(f["file_path"]); p = p if p.is_absolute() else (S / "prod/tassili/blocks_ns/lio_row100" / B / p)
        if kind == "undist": p = src / p.name
        if p.exists() and not (d / p.name).exists(): os.link(p, d / p.name); n += 1
    print(f"[camtest] {kind}: {len(list(d.glob('*.png')))} keyframes")
EOF
run() {   # tag, images dir, camera model, params, optimize_intrinsics
  local T=$1 IM=$2 M=$3 PR=$4 OPT=$5; local D=$W/$T; rm -rf $D; mkdir -p $D/sparse
  colmap feature_extractor --database_path $D/db.db --image_path $IM --ImageReader.single_camera 1 --ImageReader.camera_model $M --ImageReader.camera_params "$PR" --FeatureExtraction.use_gpu 1 > $D/colmap.log 2>&1
  colmap exhaustive_matcher --database_path $D/db.db --FeatureMatching.use_gpu 1 >> $D/colmap.log 2>&1
  $GB mapper --database_path $D/db.db --image_path $IM --output_path $D/sparse --BundleAdjustment.optimize_intrinsics $OPT --BundleAdjustment.optimize_principal_point $OPT > $D/glomap.log 2>&1
  $PY - $D $T << 'EOF' | tee -a $L
import sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model
D, T = Path(sys.argv[1]), sys.argv[2]; recs = sorted(glob.glob(str(D / "sparse/*/images.bin")))
if not recs: print(f"[camtest] {T:28s} NO RECONSTRUCTION"); sys.exit()
best = max(recs, key=lambda r: len(read_model(str(Path(r).parent), ".bin")[1])); cams, ims, pts = read_model(str(Path(best).parent), ".bin")
err = np.array([p.error for p in pts.values()]); tl = np.array([len(p.image_ids) for p in pts.values()]); c = list(cams.values())[0]
print(f"[camtest] {T:28s} registered {len(ims):3d}  points {len(pts):6d}  mean reproj {err.mean():.3f} px (median {np.median(err):.3f})  track {tl.mean():.2f}  camera {c.model} {np.round(c.params, 3).tolist()}")
EOF
}
say "=== GLOMAP camera test on $B (poses free, intrinsics FIXED unless noted)"
run copied_pinhole            $W/images_orig   PINHOLE "529.4046769303875,529.6521348953188,647.1984132364281,354.6428014457265" 0
run paul_pinhole              $W/images_orig   PINHOLE "546.8742,547.6929,638.9465,323.9356" 0
run paul_opencv_k1k2          $W/images_orig   OPENCV  "546.8742,547.6929,638.9465,323.9356,0.0238,-0.0069,0,0" 0
[ -d $W/images_undist ] && run undistorted_pinhole $W/images_undist PINHOLE "546.8742,547.6929,625.468722265625,316.73703111111115" 0
run free_opencv_from_paul     $W/images_orig   OPENCV  "546.8742,547.6929,638.9465,323.9356,0.0238,-0.0069,0,0" 1
run free_opencv_from_copied   $W/images_orig   OPENCV  "529.4046769303875,529.6521348953188,647.1984132364281,354.6428132364281,0,0,0,0" 1
say "GLOMAP CAMERA TEST DONE"
