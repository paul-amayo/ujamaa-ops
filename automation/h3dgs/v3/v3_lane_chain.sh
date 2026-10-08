#!/bin/bash
# v3_lane_chain.sh <survey root> <lane n> [stop after: extract|sfm|place|chunk] — one lane of an A300 survey (Dec ten_rows, Lindendhof)
# to a lane chunk export for recipe v3 (2026-10-08): lane window (experimental/lane_windows.json from the survey's KISS-ICP odometry)
# -> full 15 Hz left-stream extract -> SfM (GPU SIFT, exhaustive, GLOMAP, fixed ZED-conf camera; tenrows_lane_run.sh stage 1 verbatim)
# -> LO base + warp onto the LiDAR odometry + LiDAR init (stage 2 verbatim; the automation copies take LANE_SURVEY) -> H3DGS-layout
# lane project <lane>/h3dgs (tenrows_lane_prep.py chunk). No training: the v3 fleet trains it ("<survey root> <lane>/h3dgs lane 0").
set -u; S=$1; n=$2; STOP=${3:-chunk}; export LANE_SURVEY=$S; R=$S/experimental; LD=$R/lane$n; A=/home/paperspace/code/automation/tenrows
L=/home/paperspace/logs/v3_lane_$(basename $S)_lane$n.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
P310="env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib /home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10"
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python; REPO=/home/paperspace/code/hierarchical-3d-gaussians
CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap; GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
PAR="527.985,527.88,638.975,333.1835"; W=$LD/colmap; T_ALL=$(date +%s)
say "=== $(basename $S) lane $n -> $LD (stop after $STOP)"
if [ ! -e $LD/stamps.json ]; then
  read t0 t1 < <(python3 -c "import json;w=json.load(open('$R/lane_windows.json'))['$n'];print(w['t0_ms'],w['t1_ms'])")
  t=$(date +%s); $P310 $A/tenrows_lane_extract.py $t0 $t1 $LD > $L.extract 2>&1 || { say "EXTRACT FAILED (see $L.extract)"; exit 1; }
  say "extract in $(( $(date +%s)-t )) s: $(tail -1 $L.extract | cut -c1-140)"
fi
[ "$STOP" = extract ] && exit 0
if [ ! -e $W/sparse/0/images.bin ]; then
  t=$(date +%s); rm -rf $W; mkdir -p $W/sparse; export LD_LIBRARY_PATH=$GLD
  $CC feature_extractor --database_path $W/database.db --image_path $LD/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PAR" --FeatureExtraction.use_gpu 1 > $W/sfm.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
  $CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $W/sfm.log 2>&1 || { say "MATCH FAILED"; exit 1; }
  $GB mapper --database_path $W/database.db --image_path $LD/images --output_path $W/sparse --BundleAdjustment.optimize_intrinsics 0 --BundleAdjustment.optimize_principal_point 0 >> $W/sfm.log 2>&1 || { say "GLOMAP FAILED"; exit 1; }
  unset LD_LIBRARY_PATH; [ -d $W/sparse/0 ] || { say "no SfM model"; exit 1; }
  python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
  say "SfM in $(( $(date +%s)-t )) s: $(ls $LD/images | wc -l) frames -> $($PY -c "import sys;sys.path.insert(0,'$REPO/preprocess');from read_write_model import read_model;c,i,p=read_model('$W/sparse/0','.bin');print(len(i),'images',len(p),'points')" 2>&1 | tail -1)"
fi
[ "$STOP" = sfm ] && exit 0
$PY $A/tenrows_lane_prep.py lo $LD 2>&1 | tail -1 | tee -a $L
$PY $A/tenrows_lane_warp.py $LD $W/transforms.json --sigma ${LANE_SIGMA:-15} ${LANE_WARP_ARGS:-} 2>&1 | tail -1 | tee -a $L
[ -e $LD/transforms_ref_lo.json ] || { say "ALIGNMENT REFUSED"; exit 1; }
cp $LD/transforms_ref_lo.json $LD/transforms.json
[ -e $LD/init_lidar.ply ] || $PY $A/tenrows_lo_lidar_init.py $LD --stamps $LD/stamps.json --pad-x 6 --pad-y 6 --pad-z 4 2>&1 | grep -v Warn | tail -1 | tee -a $L
[ "$STOP" = place ] && exit 0
$PY $A/tenrows_lane_prep.py chunk $LD transforms_ref_lo.json --proj h3dgs 2>&1 | tail -1 | tee -a $L
say "=== lane $n chunk export done in $(( ($(date +%s)-T_ALL)/60 )) min: $(ls $LD/h3dgs/camera_calibration/chunks/lane/sparse/0 2>/dev/null | tr '\n' ' ')"
