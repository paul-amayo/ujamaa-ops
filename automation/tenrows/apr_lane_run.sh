#!/bin/bash
# apr_lane_run.sh <lane dir> [stages...] - the December lane recipe (tenrows_lane_run.sh, project h3dgs_e2) on April Klapmuts lane 2
# (Paul 2026-10-06: "run the lane recipe on april lane 2", "run it now"). Same camera (ZED conf 527.985/527.88 @ 638.975/333.18:
# the records give April and December the same sensor head), same SfM (GPU SIFT + exhaustive + GLOMAP, fixed camera), same warp
# (Sim(3) + LOESS sigma 15 + windowed yaw), same schedule (every 2nd frame, 60k / densify to 45k / grad 0.0075 / 8 M budget,
# hierarchy, train_post 30k, evaluator on the every-10th held-out frames + 60 training views). Differences forced by the data:
# the metric base is the April H3DGS export keyframe poses slerped to the stream (apr_lane_base.py; April has no laser_dump /
# KISS-ICP LO) and the LiDAR init comes from laser.monolithic placed through those poses (also apr_lane_base.py).
# Stages: sfm place train hierarchy post eval (default: sfm place). Run a frozen copy from /home/paperspace/logs.
# v2 (2026-10-06): LANE_PROJ=<project> (default h3dgs_e2), PREP_EXTRA="<extra chunk-prep args>" (e.g. --resample-cm 9.5, December's
# per-frame spacing; April was driven at 0.86 m/s vs 1.32).
set -u
LD=${1:?lane dir}; shift; STAGES=${*:-sfm place}; ITERS=60000; DU=45000; GRAD=0.0075; CAP=8000000; PROJ=${LANE_PROJ:-h3dgs_e2}; LN=apr_$(basename $LD)_${PROJ#h3dgs_}
REPO=/home/paperspace/code/hierarchical-3d-gaussians; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap; GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
L=/home/paperspace/logs/tenrows_lane_$LN.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
PAR="527.985,527.88,638.975,333.1835"; W=$LD/colmap; P=$LD/$PROJ; CH=$P/camera_calibration/chunks/lane; T=$P/output/trained_chunks/lane; TL=/home/paperspace/logs/tenrows_lane_${LN}_train.log
say "=== April lane $(basename $LD) project $PROJ stages [$STAGES]: $(ls $LD/images | wc -l) frames"
for st in $STAGES; do case $st in
sfm)
  if [ ! -e $W/sparse/0/images.bin ]; then
    t0=$(date +%s); rm -rf $W; mkdir -p $W/sparse; export LD_LIBRARY_PATH=$GLD
    $CC feature_extractor --database_path $W/database.db --image_path $LD/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PAR" --FeatureExtraction.use_gpu 1 > $W/sfm.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
    $CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $W/sfm.log 2>&1 || { say "MATCH FAILED"; exit 1; }
    $GB mapper --database_path $W/database.db --image_path $LD/images --output_path $W/sparse --BundleAdjustment.optimize_intrinsics 0 --BundleAdjustment.optimize_principal_point 0 >> $W/sfm.log 2>&1 || { say "GLOMAP FAILED"; exit 1; }
    unset LD_LIBRARY_PATH; [ -d $W/sparse/0 ] || { say "no SfM model"; exit 1; }
    python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
    say "SfM in $(( $(date +%s)-t0 ))s: $($PY -c "import sys;sys.path.insert(0,'$REPO/preprocess');from read_write_model import read_model;c,i,p=read_model('$W/sparse/0','.bin');import numpy as np;print(len(i),'images',len(p),'points, reproj %.2f px'%np.mean([q.error for q in p.values()]))")"
  else say "SfM reused ($W/sparse/0)"; fi ;;
place)
  [ -e $LD/transforms_lo.json ] && [ -e $LD/init_lidar.ply ] || { say "NO BASE (run apr_lane_base.py first)"; exit 1; }
  $PY /home/paperspace/logs/tenrows_lane_warp_v3.py $LD $W/transforms.json --sigma 15 --split-jumps 0.5 --roll-from-lo 10 2>&1 | tail -1 | tee -a $L
  [ -e $LD/transforms_ref_lo.json ] || { say "ALIGNMENT REFUSED"; exit 1; }
  cp $LD/transforms_ref_lo.json $LD/transforms.json
  $PY -c "import json;p='$LD/transforms.json';J=json.load(open(p));J['ply_file_path']='init_lidar.ply';json.dump(J,open(p,'w'),indent=1)"; say "LiDAR init $LD/init_lidar.ply"
  $PY /home/paperspace/logs/apr_lane_prep.py chunk $LD transforms_ref_lo.json --proj $PROJ --every 2 ${PREP_EXTRA:-} 2>&1 | tail -1 | tee -a $L ;;
train)
  mkdir -p $T; cd $REPO || exit 1; export CUDA_HOME=/home/paperspace/code/_cuda12 H3DGS_MAX_GAUSSIANS=$CAP
  if [ -z "$(ls -d $T/point_cloud/iteration_*/point_cloud.{ply,bin} 2>/dev/null)" ]; then
    t0=$(date +%s)
    $PY -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images --iterations $ITERS --position_lr_max_steps $ITERS --densify_until_iter $DU --densify_grad_threshold $GRAD \
      --exposure_lr_init 0.0 --eval -s $CH --model_path $T --bounds_file $CH > $TL 2>&1
    say "train rc=$? in $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < $TL | grep -oE 'Size=[0-9]+' | tail -1) $(grep -a 'budget' $TL | head -1 | cut -c1-80)"
    [ -n "$(ls -d $T/point_cloud/iteration_*/point_cloud.{ply,bin} 2>/dev/null)" ] || { say "NO POINT CLOUD"; exit 1; }
  fi ;;
hierarchy)
  cd $REPO || exit 1
  if [ ! -e $T/hierarchy.hier ]; then
    PLY=$(ls -d $T/point_cloud/iteration_* | sort -t_ -k2 -n | tail -1)/point_cloud.ply
    submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH $T >> $TL 2>&1
    [ -e $T/hierarchy.hier ] || { say "NO HIERARCHY"; exit 1; }
    say "hierarchy $(stat -c %s $T/hierarchy.hier) bytes"
  fi ;;
post)
  cd $REPO || exit 1; export CUDA_HOME=/home/paperspace/code/_cuda12
  if [ ! -e $T/hierarchy.hier_opt ]; then
    t0=$(date +%s)
    $PY -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 30000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i ../../rectified/images --exposure_lr_init 0.0 --eval -s $CH --model_path $T --hierarchy $T/hierarchy.hier >> $TL 2>&1
    say "post-opt rc=$? in $(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt')"
  fi ;;
eval)
  export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/code/_cuda12/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/paperspace/code/_cuda12
  $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 3 --save 10 --out output/eval_lane > /home/paperspace/logs/tenrows_lane_${LN}_eval.log 2>&1
  $PY /home/paperspace/logs/h3dgs_eval_chunk.py $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 3 --train_sample 60 --save 0 --out output/eval_lane_train > /home/paperspace/logs/tenrows_lane_${LN}_train_eval.log 2>&1
  say "HELD-OUT: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/tenrows_lane_${LN}_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | tr '\n' ' | ')"
  say "TRAINING: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/tenrows_lane_${LN}_train_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | tr '\n' ' | ')" ;;
*) say "unknown stage $st"; exit 1 ;;
esac; done
say "=== April lane $(basename $LD) stages [$STAGES] done"
