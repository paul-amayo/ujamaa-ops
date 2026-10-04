#!/bin/bash
# Fruit-densify 05 chunk 1_0 natively (Paul 2026-10-04: "densify the chunk then"). Sibling project h3dgs_expo_fruitd:
# camera_calibration + scaffold symlinked from h3dgs_expo; trained_chunks/1_0 = the fruit fine-tune of the trained chunk.
#  1 fine-tune (train_single_fruit.py, 6000 its, fruit boost 5, split 4 cm..0.5 m)  2 GaussianHierarchyCreator
#  3 train_post (stock 15000-its post-opt; the chunk's original 294 trained exposures restored first)
#  4 native cell chunk_1_0_sam3_fruitd (native_sam3_cell.sh)  5 fruit best-containment score vs the baseline
set -uo pipefail
S=/home/paperspace/data/citrus_all/05_13D_Jackal; P=$S/experimental/h3dgs_expo; O=$S/experimental/h3dgs_expo_fruitd; CN=1_0
CH=$P/camera_calibration/chunks; SC=$P/output/scaffold/point_cloud/iteration_30000; T=$O/output/trained_chunks/$CN
REPO=/home/paperspace/code/hierarchical-3d-gaussians; FD=/home/paperspace/code/automation/h3dgs/fruit_densify; NAT=/home/paperspace/code/automation/h3dgs/native
SUP=$S/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only; NB=$S/experimental/h3dgs_native/chunk_1_0_sam3_fruitd
L=/home/paperspace/logs/fruit_densify_05.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:$PATH CUDA_HOME=/home/paperspace/code/_cuda12; export PATH=$CUDA_HOME/bin:$PATH
mkdir -p $T $O/output; [ -e $O/camera_calibration ] || ln -s $P/camera_calibration $O/camera_calibration; [ -e $O/output/scaffold ] || ln -s $P/output/scaffold $O/output/scaffold
cd $REPO
STAGES=${*:-ft hier post cell score}
for ST in $STAGES; do say "=== $ST"; case $ST in
ft)  t0=$(date +%s)
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $FD/train_single_fruit.py --chunk_trained $P/output/trained_chunks/$CN --load_iter 60000 --fruit_sup $SUP \
       -s $CH/$CN --model_path $T -i ../../rectified/images --scaffold_file $SC --eval --iterations 6000 --densification_interval 300 --densify_from_iter 0 \
       --densify_until_iter 6001 --opacity_reset_interval 1000000 --position_lr_init 0.000002 --position_lr_final 0.0000002 --position_lr_max_steps 6000 \
       --exposure_lr_init 0.0 --exposure_lr_final 0.0 --fruit_boost 5 --fruit_split_m 0.04 --fruit_max_m 0.5 --max_gaussians 15000000 > $T/fruit_ft.log 2>&1
     say "fine-tune rc=$? in $(( $(date +%s)-t0 ))s: $(grep -a '^\[fruit-ft\] \(loaded\|fruit masks\|saved\)' $T/fruit_ft.log | tr '\n' ' ' | cut -c1-400)"
     grep -a "^\[fruit-ft\] it " $T/fruit_ft.log | tail -n 3 | tee -a $L
     cp $P/output/trained_chunks/$CN/exposure.json $T/exposure.json   # the chunk's 294 trained exposures (held-out views stay on the mean)
     [ -e $T/point_cloud/iteration_6000/done_xyz.pt ] || [ -e $T/point_cloud/iteration_6000/point_cloud.bin ] || { say "fine-tune produced no point cloud"; exit 1; } ;;
hier) t0=$(date +%s); submodules/gaussianhierarchy/build/GaussianHierarchyCreator $T/point_cloud/iteration_6000/point_cloud.ply $CH/$CN $T $SC > $T/hier.log 2>&1
     [ -e $T/hierarchy.hier ] && say "hierarchy in $(( $(date +%s)-t0 ))s ($(du -h $T/hierarchy.hier | cut -f1))" || { say "no hierarchy (see $T/hier.log)"; exit 1; } ;;
post) t0=$(date +%s); PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 \
       --save_iterations -1 -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH/$CN --model_path $T --hierarchy $T/hierarchy.hier > $T/post.log 2>&1
     [ -e $T/hierarchy.hier_opt ] && say "post-opt in $(( $(date +%s)-t0 ))s" || { say "no hier_opt (see $T/post.log)"; exit 1; } ;;
cell) [ -d $NB ] && mv $NB ${NB}_old_$(date +%H%M)
     SVN=05_13D_Jackal PROJ=$O CN=$CN OUT=$NB bash $NAT/native_sam3_cell.sh > /home/paperspace/logs/fruit_densify_05_cell.log 2>&1
     (cd /home/paperspace/code/nerf_new && pixi run python $NAT/build_text_bank.py --manifest $NB/supervision/trees_only/manifest.json --hierarchy-json $S/prod/bateleur/scene_graph/marker_hierarchy.json --out $NB/text_bank.npz 2>&1 | grep -a "text-bank\|Error" | tee -a $L)
     [ -f $NB/features_B_bg2share.bin ] && say "cell: $(python3 -c "import json; d=json.load(open('$NB/features_B_bg2share.bin.json')); print(d['assigned'], 'of', d['n'], 'nodes seeded')")" || { say "cell FAILED"; exit 1; } ;;
score) cd /home/paperspace/code && python automation/h3dgs/fruit_densify/fruit_bc_score.py $O $NB fruitd 2>&1 | grep -a "fruit-bc\|Error\|Traceback" | tee -a $L ;;
esac; done
say "=== done: $STAGES"
