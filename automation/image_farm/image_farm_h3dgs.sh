#!/bin/bash
# image_farm_h3dgs.sh <segment dir> [ITERS=60000] [DENSIFY_UNTIL=45000] [GRAD=0.0075] [CAP=4000000] — H3DGS for one phone
# segment, single chunk (Paul, 2026-09-30: "hd3gs is demo standard, anything less than 25 db should not be shown").
# The Klapmuts lane-2 recipe (automation/tenrows/tenrows_lane_run.sh steps 3-4; lane 2 half-view reached a 29.7 dB
# training-view median): train_single long densifying schedule under a gaussian budget (arm E: 4 M costs no held-out
# quality), no depth prior, no scaffold -> hierarchy -> train_post 30k -> the chunk evaluator on held-out (every 10th)
# and on a training-view sample. Project: <segment>/${IF_H3DGS_PROJ:-h3dgs}. Log: ~/logs/if_h3dgs_<segment>.log
set -u
S=$(readlink -f "${1:?segment dir}"); ITERS=${2:-60000}; DU=${3:-45000}; GRAD=${4:-0.0075}; CAP=${5:-4000000}; PROJ=${IF_H3DGS_PROJ:-h3dgs}
N=$(basename $S); P=$S/$PROJ; CH=$P/camera_calibration/chunks/lane; T=$P/output/trained_chunks/lane
REPO=/home/paperspace/code/hierarchical-3d-gaussians; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
L=/home/paperspace/logs/if_h3dgs_$N.log; TL=/home/paperspace/logs/if_h3dgs_${N}_train.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== $N -> $P: iters $ITERS densify-until $DU grad $GRAD budget $CAP"
$PY /home/paperspace/code/automation/image_farm/image_farm_h3dgs_prep.py $S --proj $PROJ 2>&1 | tail -1 | tee -a $L
[ -e $CH/center.txt ] || { say "PREP FAILED"; exit 1; }
mkdir -p $T; cd $REPO || exit 1; unset CUDA_HOME; export CUDA_HOME=/home/paperspace/code/_cuda12 H3DGS_MAX_GAUSSIANS=$CAP
if [ -z "$(ls -d $T/point_cloud/iteration_*/point_cloud.{ply,bin} 2>/dev/null)" ]; then
  t0=$(date +%s)
  $PY -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images --iterations $ITERS --position_lr_max_steps $ITERS \
    --densify_until_iter $DU --densify_grad_threshold $GRAD --exposure_lr_init 0.0 --eval -s $CH --model_path $T --bounds_file $CH > $TL 2>&1
  say "train rc=$? in $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < $TL | grep -oE 'Size=[0-9]+' | tail -1)"
  [ -n "$(ls -d $T/point_cloud/iteration_*/point_cloud.{ply,bin} 2>/dev/null)" ] || { say "NO POINT CLOUD"; exit 1; }
fi
if [ ! -e $T/hierarchy.hier ]; then
  PLY=$(ls -d $T/point_cloud/iteration_* | sort -t_ -k2 -n | tail -1)/point_cloud.ply
  submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH $T >> $TL 2>&1
  [ -e $T/hierarchy.hier ] || { say "NO HIERARCHY"; exit 1; }
  say "hierarchy $(stat -c %s $T/hierarchy.hier) bytes"
fi
if [ ! -e $T/hierarchy.hier_opt ]; then
  t0=$(date +%s)
  $PY -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 30000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
    -i ../../rectified/images --exposure_lr_init 0.0 --eval -s $CH --model_path $T --hierarchy $T/hierarchy.hier >> $TL 2>&1
  say "post-opt rc=$? in $(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt')"
fi
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/code/_cuda12/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EV=/home/paperspace/logs/h3dgs_eval_chunk.py
$PY $EV $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 --save 8 --out output/eval_lane > /home/paperspace/logs/if_h3dgs_${N}_eval.log 2>&1
$PY $EV $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 --train_sample 40 --save 8 --out output/eval_lane_train > /home/paperspace/logs/if_h3dgs_${N}_train_eval.log 2>&1
say "HELD-OUT: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/if_h3dgs_${N}_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | head -1)"
say "TRAINING: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/if_h3dgs_${N}_train_eval.log | grep -oE 'tau [0-9]+: [0-9]+ views.*full-frame PSNR mean [0-9.]+ median [0-9.]+' | head -1)"
say "=== $N DONE"
