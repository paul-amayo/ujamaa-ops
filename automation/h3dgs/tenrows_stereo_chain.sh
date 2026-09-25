#!/bin/bash
# tenrows_stereo_chain.sh — ten_rows chunk 1_1 trained with the ZED RIGHT camera added (stereo experiment, 2026-09-25).
# Base = the rgb_zed survey (colour-fixed PNGs, ZED-conf camera, LiDAR seed, its scaffold, DA-V2 depth on the left
# frames). Right frames: poses = median rig transform from tenrows_stereo_register.sh, DA-V2 depth PNGs but no
# depth_params (no depth loss), held-out timestamps excluded on both cameras (test.txt). Recipe = the survey's
# train_single -> GaussianHierarchyCreator -> train_post, then the compact evaluator on the chunk (tau 3, sky-masked,
# scaffold fill outside the cell) over the same left held-out views the survey scores. A/B partner = the survey's own
# chunk 1_1 (left only), evaluated the same way once it exists.
set -u
P=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_rgb_zed
X=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_rgb_zed_stereo11
SM=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/stereo_reg_11/stereo_model
RK=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/stereo_right_kf
REPO=/home/paperspace/code/hierarchical-3d-gaussians
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; export CUDA_HOME=/home/paperspace/code/_cuda12
L=/home/paperspace/logs/tenrows_stereo_chain.log; FT=/home/paperspace/logs/tenrows_stereo_train.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
CC=$X/camera_calibration; CH=$CC/chunks/1_1; SC=$X/output/scaffold/point_cloud/iteration_30000; T=$X/output/trained_chunks/1_1
[ -e $SM/sparse/0/images.bin ] || { say "no stereo chunk model at $SM"; exit 1; }
mkdir -p $CC/rectified/images $CC/rectified/depths $CH $CC/aligned/sparse/0 $SC $T
# 1. project: the chunk's left keyframes (+ depths), the right keyframes, the stereo chunk model, the survey scaffold, aligned camera + test list
for n in $(cat /home/paperspace/logs/tenrows_chunk11_names.txt); do
  ln -f $P/camera_calibration/rectified/images/$n $CC/rectified/images/$n; ln -f $P/camera_calibration/rectified/depths/$n $CC/rectified/depths/$n
  r=${n%.png}_R.png; [ -e $RK/$r ] && ln -f $RK/$r $CC/rectified/images/$r
done
rm -rf $CH/sparse; cp -r $SM/sparse $CH/; cp $P/camera_calibration/chunks/1_1/center.txt $P/camera_calibration/chunks/1_1/extent.txt $CH/
ln -f $P/output/scaffold/point_cloud/iteration_30000/point_cloud.ply $SC/point_cloud.ply; ln -f $P/output/scaffold/point_cloud/iteration_30000/pc_info.txt $SC/pc_info.txt
for f in cameras.bin images.bin test.txt; do ln -f $P/camera_calibration/aligned/sparse/0/$f $CC/aligned/sparse/0/$f; done; cp $P/export_meta.json $X/
say "project: $(ls $CC/rectified/images | wc -l) images ($(ls $CC/rectified/images | grep -c _R) right), $(ls $CC/rectified/depths | wc -l) depths, chunk model $(python -c "import sys;sys.path.insert(0,'$REPO/preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/sparse/0/images.bin')))") images, test.txt $(wc -l < $CH/sparse/0/test.txt) names"
# 2. DA-V2 inverse depth for the right frames (the survey's depth script; skips frames that have one)
if [ "$(ls $CC/rectified/depths | wc -l)" -lt "$(ls $CC/rectified/images | wc -l)" ]; then
  t0=$(date +%s); H3DGS_PROJ=$X ${H3DGS_DEPTH_PY:-/home/paperspace/envs/hfeval_ft/bin/python} /home/paperspace/logs/h3dgs_depth_05.py > /home/paperspace/logs/tenrows_stereo_depth.log 2>&1 || { say "DEPTH FAILED"; exit 1; }
  say "right depths in $(( $(date +%s)-t0 ))s ($(ls $CC/rectified/depths | wc -l) total)"
fi
# 3. train (survey recipe) -> hierarchy -> post-opt
cd $REPO || exit 1
if [ ! -e $T/hierarchy.hier_opt ] && [ -z "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ]; then
  t0=$(date +%s)
  python -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images -d ../../rectified/depths --scaffold_file $SC --skybox_locked \
    --exposure_lr_init 0.0 --eval -s $CH --model_path $T --bounds_file $CH ${H3DGS_TRAIN_EXTRA:-} >> $FT 2>&1
  say "train chunk 1_1 stereo rc=$? wall=$(( $(date +%s)-t0 ))s"
  [ -n "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ] || { say "NO POINT CLOUD"; exit 1; }
fi
if [ ! -e $T/hierarchy.hier ] && [ ! -e $T/hierarchy.hier_opt ]; then
  PLY=$(ls -d $T/point_cloud/iteration_* | sort -t_ -k2 -n | tail -1)/point_cloud.ply
  submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH $T $SC >> $FT 2>&1
  [ -e $T/hierarchy.hier ] || { say "NO HIERARCHY"; exit 1; }
fi
if [ ! -e $T/hierarchy.hier_opt ]; then
  t0=$(date +%s)
  python -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
    -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH --model_path $T --hierarchy $T/hierarchy.hier ${H3DGS_POST_EXTRA:-} >> $FT 2>&1
  say "post-opt rc=$? wall=$(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt')"
  [ -e $T/hierarchy.hier_opt ] || exit 1
fi
# 4. held-out eval: the chunk's left held-out views inside its cell (aligned test.txt has left names only), tau 3, sky-masked
t0=$(date +%s)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/paperspace/logs/h3dgs_eval_compact.py $X --hier output/trained_chunks/1_1/hierarchy.hier_opt --only_chunk 1_1 --fill --taus 3 --out output/eval_11 > /home/paperspace/logs/tenrows_stereo_eval.log 2>&1
say "eval (stereo chunk 1_1) in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[eval\] tau" /home/paperspace/logs/tenrows_stereo_eval.log | tr '\n' ' ')"
say "=== stereo chain DONE"
