#!/bin/bash
# H3DGS on 05_13D — training chain. Waits for the preprocessing chain, then runs the reference
# pipeline end to end (coarse scaffold -> per-chunk training -> hierarchy -> post-opt -> merge) with
# exposure optimisation OFF and the held-out split ON (apples-to-apples with the per-block fleet),
# then renders the held-out views from the merged hierarchy.
L=/home/paperspace/logs/h3dgs_train.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians
PROJ=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH
cd $REPO
until grep -q "PREP DONE" /home/paperspace/logs/h3dgs_prep.log 2>/dev/null; do sleep 60; done
say "prep done ($(grep -c FAILED /home/paperspace/logs/h3dgs_prep.log) FAILED lines); chunks: $(ls $PROJ/camera_calibration/chunks | tr '\n' ' ')"
say "GPU before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
t0=$(date +%s)
python scripts/full_train.py --project_dir $PROJ --extra_training_args "--exposure_lr_init 0.0 --eval" > /home/paperspace/logs/h3dgs_full_train.log 2>&1
rc=$?
say "full_train rc=$rc wall=$(( $(date +%s)-t0 ))s"
ls -la $PROJ/output/merged.hier 2>&1 | tee -a $L
[ -f $PROJ/output/merged.hier ] || { say "NO MERGED HIERARCHY"; exit 1; }
t0=$(date +%s)
python render_hierarchy.py -s $PROJ/camera_calibration/aligned -i ../rectified/images --model_path $PROJ/output --hierarchy $PROJ/output/merged.hier \
  --out_dir $PROJ/output/renders --eval --scaffold_file $PROJ/output/scaffold/point_cloud/iteration_30000 --taus 0 3 6 > /home/paperspace/logs/h3dgs_render_eval.log 2>&1
grep -aE "tau|PSNR" /home/paperspace/logs/h3dgs_render_eval.log | tee -a $L
say "held-out render in $(( $(date +%s)-t0 ))s"
say "H3DGS TRAIN CHAIN DONE"
