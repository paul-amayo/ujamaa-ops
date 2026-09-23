#!/bin/bash
# Single-chunk ablation for the Klapmuts quality question: retrain ONE chunk with a recipe variant, build its
# hierarchy (no post-opt), and score its own held-out views with the compact renderer (tau 0 = the trained
# gaussians). Variants are passed as extra train_single args.
#   usage: h3dgs_ablate_chunk.sh <proj> <chunk> <tag> [extra train_single args...]
# Exposure variants train with --train_test_exp (exposure fitted on the LEFT half of test images) and are scored
# on the RIGHT half with the fitted exposure applied — the H3DGS evaluation protocol.
PROJ=${1:?proj}; C=${2:?chunk}; TAG=${3:?tag}; shift 3; EXTRA="$*"
L=/home/paperspace/logs/h3dgs_ablate.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians; CH=$PROJ/camera_calibration/chunks; SC=$PROJ/output/scaffold/point_cloud/iteration_30000
T=$PROJ/output/ablation/$TAG/$C; mkdir -p $T
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:$PATH; cd $REPO
say "=== ablation $TAG on $(basename $PROJ | cut -c1-12)/$C: $EXTRA"
t0=$(date +%s)
DEPTH="-d ../../rectified/depths"; case " $EXTRA " in *" --no-depth "*) DEPTH=""; EXTRA=${EXTRA//--no-depth/};; esac
python -u train_single.py --save_iterations -1 -i ../../rectified/images $DEPTH --scaffold_file $SC --skybox_locked --eval \
  -s $CH/$C --model_path $T --bounds_file $CH/$C $EXTRA > $T/train.log 2>&1
say "train rc=$? in $(( $(date +%s)-t0 ))s"
[ -e $T/point_cloud/iteration_30000/point_cloud.ply ] || { say "$TAG: no point cloud ($(grep -aE 'Error|error' $T/train.log | tail -1 | cut -c1-120))"; exit 1; }
submodules/gaussianhierarchy/build/GaussianHierarchyCreator $T/point_cloud/iteration_30000/point_cloud.ply $CH/$C $T $SC >> $T/train.log 2>&1
[ -e $T/hierarchy.hier ] || { say "$TAG: no hierarchy"; exit 1; }
EX=""; case "$EXTRA" in *train_test_exp*) EX="--exposure_json $T/exposure.json --right_half";; esac
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/paperspace/logs/h3dgs_eval_chunk.py $PROJ --hier output/ablation/$TAG/$C/hierarchy.hier --taus 0 --only_chunk $C --out output/ablation/$TAG/eval_$C --save 12 $EX > $T/eval.log 2>&1
say "$TAG $C: $(grep -aE '^\[eval\] tau' $T/eval.log | cut -c1-150) (total $(( $(date +%s)-t0 ))s)"
rm -rf $T/point_cloud
