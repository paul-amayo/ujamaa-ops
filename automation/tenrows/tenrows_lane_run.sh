#!/bin/bash
# tenrows_lane_run.sh <lane dir> [ITERS=60000] [DENSIFY_UNTIL=45000] [GRAD=0.0075] — one ten_rows lane, full 15 Hz stream,
# overtrained (Paul, 2026-09-26): SfM on all frames (GPU SIFT + exhaustive + GLOMAP, fixed ZED-conf camera), hybrid
# Sim(3) onto the LiDAR odometry (metric), LiDAR init from the raw scans at every frame, H3DGS train_single with a long
# densifying schedule (no depth prior, no scaffold), hierarchy, train_post 30k, then the per-chunk evaluator on the
# lane's own every-10th held-out frames + a training-view sample + renders.
set -u
LD=${1:?lane dir}; ITERS=${2:-60000}; DU=${3:-45000}; GRAD=${4:-0.0075}
REPO=/home/paperspace/code/hierarchical-3d-gaussians; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap; GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
L=/home/paperspace/logs/tenrows_lane_$(basename $LD).log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
PAR="527.985,527.88,638.975,333.1835"; W=$LD/colmap
say "=== lane $(basename $LD): $(ls $LD/images | wc -l) frames; schedule iters $ITERS densify-until $DU grad $GRAD"
# 1. SfM on all frames
if [ ! -e $W/sparse/0/images.bin ]; then
  t0=$(date +%s); rm -rf $W; mkdir -p $W/sparse; export LD_LIBRARY_PATH=$GLD
  $CC feature_extractor --database_path $W/database.db --image_path $LD/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PAR" --FeatureExtraction.use_gpu 1 > $W/sfm.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
  $CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $W/sfm.log 2>&1 || { say "MATCH FAILED"; exit 1; }
  $GB mapper --database_path $W/database.db --image_path $LD/images --output_path $W/sparse --BundleAdjustment.optimize_intrinsics 0 --BundleAdjustment.optimize_principal_point 0 >> $W/sfm.log 2>&1 || { say "GLOMAP FAILED"; exit 1; }
  unset LD_LIBRARY_PATH; [ -d $W/sparse/0 ] || { say "no SfM model"; exit 1; }
  python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
  say "SfM in $(( $(date +%s)-t0 ))s: $($PY -c "import sys;sys.path.insert(0,'$REPO/preprocess');from read_write_model import read_model;c,i,p=read_model('$W/sparse/0','.bin');import numpy as np;print(len(i),'images',len(p),'points, reproj %.2f px'%np.mean([q.error for q in p.values()]))")"
fi
# 2. LO base + hybrid Sim(3) placement + LiDAR init
$PY /home/paperspace/logs/tenrows_lane_prep.py lo $LD 2>&1 | tail -1 | tee -a $L
python3 /home/paperspace/logs/klapmuts_apply_refine_orient.py $LD $W transforms_lo.json transforms_ref_lo.json 2>&1 | tail -1 | tee -a $L
[ -e $LD/transforms_ref_lo.json ] || { say "ALIGNMENT REFUSED"; exit 1; }
cp $LD/transforms_ref_lo.json $LD/transforms.json
$PY /home/paperspace/logs/tenrows_lo_lidar_init.py $LD --stamps $LD/stamps.json --pad-x 6 --pad-y 6 --pad-z 4 2>&1 | grep -v Warn | tail -1 | tee -a $L
$PY /home/paperspace/logs/tenrows_lane_prep.py chunk $LD transforms_ref_lo.json 2>&1 | tail -1 | tee -a $L
# 3. H3DGS: train_single (long, densifying), hierarchy, train_post
P=$LD/h3dgs; CH=$P/camera_calibration/chunks/lane; T=$P/output/trained_chunks/lane; mkdir -p $T; cd $REPO || exit 1; unset CUDA_HOME
export CUDA_HOME=/home/paperspace/code/_cuda12
if [ -z "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ]; then
  t0=$(date +%s)
  $PY -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images --iterations $ITERS --position_lr_max_steps $ITERS --densify_until_iter $DU --densify_grad_threshold $GRAD \
    --exposure_lr_init 0.0 --eval -s $CH --model_path $T --bounds_file $CH > /home/paperspace/logs/tenrows_lane_$(basename $LD)_train.log 2>&1
  say "train rc=$? in $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < /home/paperspace/logs/tenrows_lane_$(basename $LD)_train.log | grep -oE 'Size=[0-9]+' | tail -1)"
  [ -n "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ] || { say "NO POINT CLOUD"; exit 1; }
fi
if [ ! -e $T/hierarchy.hier ]; then
  PLY=$(ls -d $T/point_cloud/iteration_* | sort -t_ -k2 -n | tail -1)/point_cloud.ply
  submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH $T >> /home/paperspace/logs/tenrows_lane_$(basename $LD)_train.log 2>&1
  [ -e $T/hierarchy.hier ] || { say "NO HIERARCHY"; exit 1; }
fi
if [ ! -e $T/hierarchy.hier_opt ]; then
  t0=$(date +%s)
  $PY -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 30000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i ../../rectified/images --exposure_lr_init 0.0 --eval -s $CH --model_path $T --hierarchy $T/hierarchy.hier >> /home/paperspace/logs/tenrows_lane_$(basename $LD)_train.log 2>&1
  say "post-opt rc=$? in $(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt')"
fi
# 4. eval: held-out (every 10th) and training views, renders
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/code/_cuda12/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 3 --save 10 --out output/eval_lane > /home/paperspace/logs/tenrows_lane_$(basename $LD)_eval.log 2>&1
$PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 3 --train_sample 60 --save 0 --out output/eval_lane_train > /home/paperspace/logs/tenrows_lane_$(basename $LD)_train_eval.log 2>&1
say "HELD-OUT: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/tenrows_lane_$(basename $LD)_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | tr '\n' ' | ')"
say "TRAINING: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/tenrows_lane_$(basename $LD)_train_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | tr '\n' ' | ')"
say "=== lane $(basename $LD) DONE"
