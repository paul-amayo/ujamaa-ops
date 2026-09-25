#!/bin/bash
# tenrows_stereo_register.sh — register the ZED right-camera keyframes of ten_rows chunk 1_1 against the chunk's LEFT
# model (left poses fixed, shared PINHOLE camera) with the survey's COLMAP 3.14 (GPU SIFT), measure the left->right rig
# transform per frame, and write the stereo chunk model (left + right, right poses = median rig transform).
# Inputs: rgb_zed chunk 1_1, right PNGs from tenrows_right_kf_extract.py. Output: <W>/stereo_model/sparse/0 + rig.json.
set -u
P=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_rgb_zed
CH=$P/camera_calibration/chunks/1_1; IMGS=$P/camera_calibration/rectified/images
RK=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/stereo_right_kf
W=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/stereo_reg_11
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
L=/home/paperspace/logs/tenrows_stereo_register.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
mkdir -p $W/images $W/left_model $W/tri $W/reg
for n in $(cat /home/paperspace/logs/tenrows_chunk11_names.txt); do ln -f $IMGS/$n $W/images/$n; r=${n%.png}_R.png; [ -e $RK/$r ] && ln -f $RK/$r $W/images/$r; done
say "images: $(ls $W/images | wc -l) ($(ls $W/images | grep -c _R) right)"
PARAMS=$($PY -c "import sys;sys.path.insert(0,'/home/paperspace/code/hierarchical-3d-gaussians/preprocess');from read_write_model import read_cameras_binary as r;c=r('$CH/sparse/0/cameras.bin')[1];print(','.join(str(x) for x in c.params))")
say "camera params $PARAMS"
rm -f $W/db.db; t0=$(date +%s)
colmap feature_extractor --database_path $W/db.db --image_path $W/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 1 > $W/colmap.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
colmap sequential_matcher --database_path $W/db.db --SequentialMatching.overlap 12 --FeatureMatching.use_gpu 1 >> $W/colmap.log 2>&1 || { say "MATCHING FAILED"; exit 1; }
say "features + sequential matches (overlap 12, L/R interleaved) in $(( $(date +%s)-t0 ))s"
$PY /home/paperspace/logs/tenrows_stereo_models.py left $W/db.db $CH/sparse/0 $W/left_model 2>&1 | tee -a $L || { say "LEFT MODEL FAILED"; exit 1; }
t0=$(date +%s)
colmap point_triangulator --database_path $W/db.db --image_path $W/images --input_path $W/left_model --output_path $W/tri --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.fix_existing_frames 1 >> $W/colmap.log 2>&1 || { say "TRIANGULATION FAILED"; exit 1; }
say "left triangulated in $(( $(date +%s)-t0 ))s: $($PY -c "import sys;sys.path.insert(0,'/home/paperspace/code/hierarchical-3d-gaussians/preprocess');from read_write_model import read_model;c,i,p=read_model('$W/tri','.bin');print(len(i),'images',len(p),'points')")"
t0=$(date +%s)
colmap mapper --database_path $W/db.db --image_path $W/images --input_path $W/tri --output_path $W/reg --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.fix_existing_frames 1 >> $W/colmap.log 2>&1 || { say "REGISTRATION FAILED"; exit 1; }
say "right images registered in $(( $(date +%s)-t0 ))s"
$PY /home/paperspace/logs/tenrows_stereo_models.py rig $W/reg $CH/sparse/0 $W/stereo_model 2>&1 | tee -a $L
