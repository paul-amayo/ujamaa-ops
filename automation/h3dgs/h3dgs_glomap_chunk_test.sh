#!/bin/bash
# Direct test of an improved trajectory: build a sibling project on the GLOMAP-aligned poses (same images, depths
# and scaffold hardlinked), cut the same chunks with poses fixed, and retrain ONE chunk with the baseline recipe;
# its held-out PSNR is directly comparable with the same chunk trained on the global-BA poses.
#   usage: h3dgs_glomap_chunk_test.sh <proj> <chunk>
SRC=${1:?proj}; C=${2:?chunk}; DST=${SRC}_glomap; CC=$DST/camera_calibration
L=/home/paperspace/logs/h3dgs_glomap_traj.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; cd $REPO
[ -e $SRC/camera_calibration/glomap_aligned/sparse/0/images.bin ] || { say "no GLOMAP-aligned model in $SRC"; exit 1; }
mkdir -p $CC/rectified $CC/aligned/sparse/0 $DST/output/scaffold
cp $SRC/export_meta.json $DST/
[ -e $CC/rectified/images ] || cp -al $SRC/camera_calibration/rectified/images $CC/rectified/images   # hardlinks
[ -e $CC/rectified/depths ] || cp -al $SRC/camera_calibration/rectified/depths $CC/rectified/depths
[ -e $DST/output/scaffold/point_cloud ] || cp -al $SRC/output/scaffold/point_cloud $DST/output/scaffold/point_cloud
cp $SRC/camera_calibration/glomap_aligned/sparse/0/*.bin $SRC/camera_calibration/glomap_aligned/sparse/0/test.txt $CC/aligned/sparse/0/
echo globalba > $CC/aligned/RECIPE
say "=== GLOMAP-pose chunk test: $DST chunk $C"
if [ ! -e $CC/chunks/$C/sparse/0/depth_params.json ]; then
  t0=$(date +%s); rm -rf $CC/raw_chunks $CC/chunks
  $PY preprocess/make_chunk.py --base_dir $CC/aligned/sparse/0 --images_dir $CC/rectified/images --chunk_size 30 --lapla_thresh 0 --min_n_cams 50 --max_n_cams 1500 --output_path $CC/raw_chunks >> $L 2>&1
  for RC in $(ls $CC/raw_chunks); do
    $PY preprocess/prepare_chunk.py --raw_chunk $CC/raw_chunks/$RC --out_chunk $CC/chunks/$RC --images_dir $CC/rectified/images --skip_bundle_adjustment > /home/paperspace/logs/h3dgs_glomap_chunk_$RC.log 2>&1 || say "chunk $RC TRIANGULATION FAILED"
    rm -rf $CC/raw_chunks/$RC/bundle_adjustment/images $CC/raw_chunks/$RC/bundle_adjustment/stereo $CC/raw_chunks/$RC/bundle_adjustment/database.db*
  done
  $PY preprocess/make_chunks_depth_scale.py --chunks_dir $CC/chunks --depths_dir $CC/rectified/depths >> $L 2>&1
  $PY preprocess/copy_file_to_chunks.py --file_path $CC/aligned/sparse/0/test.txt --chunks_path $CC/chunks >> $L 2>&1
  echo globalba > $CC/chunks/RECIPE
  say "chunks on GLOMAP poses in $(( $(date +%s)-t0 ))s: $(for c in $(ls $CC/chunks); do echo -n "$c=$($PY -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CC/chunks/$c/sparse/0/images.bin')))") "; done)"
fi
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 14000 ]; do sleep 30; done
/home/paperspace/logs/h3dgs_ablate_chunk.sh $DST $C glomap_poses --exposure_lr_init 0.0 2>&1 | grep -aE "^\[" | tail -1 | cut -c1-220
say "GLOMAP CHUNK TEST DONE"
