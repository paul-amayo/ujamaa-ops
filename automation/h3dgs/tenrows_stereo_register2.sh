#!/bin/bash
# tenrows_stereo_register2.sh — second attempt at the right-camera registration: the incremental mapper crashed at the
# end of its global BA (std::out_of_range in Reconstruction::Image, COLMAP 3.14-dev, model never written). Register the
# 742 right images into the triangulated LEFT model with image_registrator (PnP + local refinement, existing poses
# untouched), then the rig step of tenrows_stereo_models.py. Reuses the database, features and matches of
# tenrows_stereo_register.sh.
set -u
P=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_rgb_zed; CH=$P/camera_calibration/chunks/1_1
W=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/stereo_reg_11
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
L=/home/paperspace/logs/tenrows_stereo_register.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
rm -rf $W/reg2; mkdir -p $W/reg2; t0=$(date +%s)
colmap image_registrator --database_path $W/db.db --input_path $W/tri --output_path $W/reg2 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 > $W/colmap_reg2.log 2>&1 \
  || { say "IMAGE_REGISTRATOR FAILED rc=$? ($(grep -aE 'ERROR|terminate|Check failed' $W/colmap_reg2.log | tail -1 | cut -c1-160))"; exit 1; }
say "image_registrator in $(( $(date +%s)-t0 ))s: $($PY -c "import sys;sys.path.insert(0,'/home/paperspace/code/hierarchical-3d-gaussians/preprocess');from read_write_model import read_model;c,i,p=read_model('$W/reg2','.bin');print(len(i),'images',len(p),'points', sum(1 for im in i.values() if im.name.endswith('_R.png')),'right')")"
$PY /home/paperspace/logs/tenrows_stereo_models.py rig $W/reg2 $CH/sparse/0 $W/stereo_model 2>&1 | tee -a $L
