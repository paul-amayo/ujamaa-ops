#!/bin/bash
# Resume the H3DGS 05_13D training on a SHARED GPU (fleet still running): idempotent per-chunk steps
# (train_single -> hierarchy creator -> train_post), small chunks first, the two big middle chunks last;
# an OOM on a chunk is logged and skipped so a later re-run can finish it once the GPU is free.
# Merge whatever chunks completed, then render the held-out views. Outputs are checked, never redone.
L=/home/paperspace/logs/h3dgs_train.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians
PROJ=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs
OUT=$PROJ/output; CH=$PROJ/camera_calibration/chunks; SC=$OUT/scaffold/point_cloud/iteration_30000
FT=/home/paperspace/logs/h3dgs_full_train.log
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH
cd $REPO
ORDER="0_0 0_2 1_0 1_2 0_1 1_1"
say "RESUME on shared GPU ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used); order: $ORDER"
for c in $ORDER; do
  T=$OUT/trained_chunks/$c; mkdir -p $T
  if [ ! -e $T/point_cloud/iteration_30000/point_cloud.ply ]; then
    t0=$(date +%s); say "train chunk $c ($(ls $CH/$c/sparse/0 >/dev/null 2>&1 && python -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/$c/sparse/0/images.bin')))") cams)"
    python -u train_single.py --save_iterations -1 -i ../../rectified/images -d ../../rectified/depths --scaffold_file $SC --skybox_locked \
      --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --bounds_file $CH/$c >> $FT 2>&1
    say "train chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s"
    [ -e $T/point_cloud/iteration_30000/point_cloud.ply ] || { say "chunk $c: no point cloud — skipping (see h3dgs_full_train.log)"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier ]; then
    t0=$(date +%s)
    submodules/gaussianhierarchy/build/GaussianHierarchyCreator $T/point_cloud/iteration_30000/point_cloud.ply $CH/$c $T $SC >> $FT 2>&1
    say "hierarchy chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s"
    [ -e $T/hierarchy.hier ] || { say "chunk $c: no hierarchy — skipping"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier_opt ]; then
    t0=$(date +%s); say "post-opt chunk $c (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used before)"
    python -u train_post.py --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
      -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --hierarchy $T/hierarchy.hier >> $FT 2>&1
    say "post-opt chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt (OOM?)')"
  fi
done
DONE=$(for c in $ORDER; do [ -e $OUT/trained_chunks/$c/hierarchy.hier_opt ] && echo -n "$c "; done)
say "chunks complete: $DONE"
[ -n "$DONE" ] || { say "NO CHUNK COMPLETED"; exit 1; }
t0=$(date +%s)
submodules/gaussianhierarchy/build/GaussianHierarchyMerger $OUT/trained_chunks 0 $CH $OUT/merged.hier $DONE >> $FT 2>&1
say "merge rc=$? wall=$(( $(date +%s)-t0 ))s $(ls -la $OUT/merged.hier 2>/dev/null | awk '{print $5" bytes"}')"
[ -f $OUT/merged.hier ] || { say "NO MERGED HIERARCHY"; exit 1; }
t0=$(date +%s)
python render_hierarchy.py -s $PROJ/camera_calibration/aligned -i ../rectified/images --model_path $OUT --hierarchy $OUT/merged.hier \
  --out_dir $OUT/renders --eval --scaffold_file $SC --taus 0 3 6 > /home/paperspace/logs/h3dgs_render_eval.log 2>&1
grep -aE "tau|PSNR" /home/paperspace/logs/h3dgs_render_eval.log | tee -a $L
say "held-out render in $(( $(date +%s)-t0 ))s"
say "H3DGS TRAIN CHAIN DONE"
