#!/bin/bash
# Re-rectify a Klapmuts survey's keyframes with a calibrated OPENCV camera and export an H3DGS project on the
# undistorted PINHOLE images: (1) export once with the nominal camera to get the keyframe list + poses, (2) rewrite
# the camera as OPENCV(fx fy cx cy k1 k2 p1 p2), (3) colmap image_undistorter -> pinhole images + new intrinsics,
# (4) re-export the project on those images with H3DGS_INTRINSICS = the undistorter's PINHOLE camera.
#   usage: h3dgs_undistort_export.sh <survey_root> <proj_dir> "fx,fy,cx,cy,k1,k2,p1,p2"     (env: H3DGS_BLOCK_VARIANT passes through)
S=${1:?survey}; P=${2:?proj}; CAM=${3:?opencv params}
L=/home/paperspace/logs/h3dgs_undistort.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
U=$P/undistort; mkdir -p $U; TMP=$U/nominal_export
say "=== undistort+export $(basename $S) -> $P with OPENCV [$CAM]"
[ -e $TMP/export_meta.json ] || $PY /home/paperspace/logs/h3dgs_export.py $S $TMP 2>&1 | tail -1 | tee -a $L
$PY - $TMP $CAM << 'EOF' || { say "camera rewrite failed"; exit 1; }
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, write_model, Camera
T = Path(sys.argv[1]); params = np.array([float(x) for x in sys.argv[2].split(",")]); assert len(params) == 8
cams, ims, pts = read_model(str(T / "camera_calibration/poses/sparse/0"), ".bin"); c = cams[1]
out = T / "camera_calibration/opencv/sparse/0"; out.mkdir(parents=True, exist_ok=True)
write_model({1: Camera(id=1, model="OPENCV", width=c.width, height=c.height, params=params)}, ims, pts, str(out), ".bin"); print(f"[undistort] OPENCV model written for {len(ims)} images")
EOF
if [ ! -e $U/sparse/cameras.bin ]; then
  t0=$(date +%s)
  colmap image_undistorter --image_path $TMP/camera_calibration/rectified/images --input_path $TMP/camera_calibration/opencv/sparse/0 --output_path $U --output_type COLMAP --max_image_size 1280 > $U/undistorter.log 2>&1 || { say "image_undistorter FAILED (see $U/undistorter.log)"; exit 1; }
  say "undistorted $(ls $U/images | wc -l) images in $(( $(date +%s)-t0 ))s"
fi
NEWCAM=$($PY -c "
import sys; sys.path.insert(0,'/home/paperspace/code/hierarchical-3d-gaussians/preprocess'); from read_write_model import read_cameras_binary as r
c=r('$U/sparse/cameras.bin')[1]; print(','.join(str(float(x)) for x in c.params)); import sys as s; s.stderr.write(f'{c.model} {c.width}x{c.height} {[round(float(x),2) for x in c.params]}\n')")
say "undistorted PINHOLE camera: $NEWCAM"
H3DGS_IMAGES_DIR=$U/images H3DGS_INTRINSICS=$NEWCAM $PY /home/paperspace/logs/h3dgs_export.py $S $P 2>&1 | tail -2 | tee -a $L
echo "$CAM" > $P/undistort/opencv_camera.txt; say "UNDISTORT EXPORT DONE -> $P"
