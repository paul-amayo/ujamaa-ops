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
c=r('$U/sparse/cameras.bin')[1]; print(','.join(str(float(x)) for x in c.params) + f',{c.width},{c.height}')")
say "undistorted PINHOLE camera (fx,fy,cx,cy,w,h): $NEWCAM"
# the sky / foreground masks live on the original pixel grid: warp them with the same undistortion (nearest)
$PY - $S $U "$CAM" "$NEWCAM" << 'EOF' 2>&1 | tail -2 | tee -a $L
import sys, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
S, U = Path(sys.argv[1]), Path(sys.argv[2]); cam = [float(x) for x in sys.argv[3].split(",")]; new = [float(x) for x in sys.argv[4].split(",")]
K = np.array([[cam[0], 0, cam[2]], [0, cam[1], cam[3]], [0, 0, 1]]); D = np.array(cam[4:8]); Kn = np.array([[new[0], 0, new[2]], [0, new[1], new[3]], [0, 0, 1]]); W, H = int(new[4]), int(new[5])
m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Kn, (W, H), cv2.CV_32FC1)
names = set(p.name for p in (U / "images").glob("*.png"))
for kind in ("sky_masks", "fg_masks"):
    src = S / "prod/tassili" / kind
    if not src.exists(): print(f"[undistort] no {kind}"); continue
    dst = U / kind; dst.mkdir(exist_ok=True)
    def one(f):
        if (dst / f.name).exists(): return 0
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); cv2.imwrite(str(dst / f.name), cv2.remap(m, m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)); return 1
    fs = [f for f in src.glob("*.png") if f.name in names]
    with ThreadPoolExecutor(16) as ex: n = sum(ex.map(one, fs))
    print(f"[undistort] {kind}: {n} masks warped to {W}x{H} ({len(fs)} keyframes)")
EOF
SKY=""; FG=""; [ -d $U/sky_masks ] && SKY=$U/sky_masks; [ -d $U/fg_masks ] && FG=$U/fg_masks
H3DGS_IMAGES_DIR=$U/images H3DGS_INTRINSICS=$NEWCAM H3DGS_SKY_MASKS=$SKY H3DGS_FG_MASKS=$FG $PY /home/paperspace/logs/h3dgs_export.py $S $P 2>&1 | tail -2 | tee -a $L
echo "$CAM" > $P/undistort/opencv_camera.txt; say "UNDISTORT EXPORT DONE -> $P"
