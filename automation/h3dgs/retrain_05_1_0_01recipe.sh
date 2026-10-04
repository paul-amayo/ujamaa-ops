#!/bin/bash
# Retrain 05 chunk 1_0 with the 01 FORMULA (Paul 2026-10-04: "can we use citrus 01 formula to get better" -> "retrain first before
# densification"). The 01 improved recipe (logs/h3dgs_01_improved_few.sh): ALL views (the export's 1-in-10 held out), 60k its,
# densify until 45k, grad 0.0075, 12 M cap, per-image exposure 0.001 (train + post), depth prior, scaffold, bounds, skybox locked.
# Sibling project h3dgs_01recipe reuses the CURRENT Citrus B chunk (h3dgs_expo 1_0: same images, same chunk-BA poses, rectified
# images/depths, scaffold) so the demo paths and cells stay aligned; only the split (test.txt) and the recipe change.
set -uo pipefail
S=/home/paperspace/data/citrus_all/05_13D_Jackal; E=$S/experimental/h3dgs_expo; N=$S/experimental/h3dgs_01recipe; CN=1_0
CH=$N/camera_calibration/chunks; SC=$N/output/scaffold/point_cloud/iteration_30000; T=$N/output/trained_chunks/$CN
REPO=/home/paperspace/code/hierarchical-3d-gaussians; NAT=/home/paperspace/code/automation/h3dgs/native; NB=$S/experimental/h3dgs_native/chunk_1_0_sam3_01r
L=/home/paperspace/logs/retrain_05_1_0_01recipe.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:$PATH CUDA_HOME=/home/paperspace/code/_cuda12 H3DGS_MAX_GAUSSIANS=12000000; export PATH=$CUDA_HOME/bin:$PATH
STAGES=${*:-setup train hier post eval cell score}
for ST in $STAGES; do say "=== $ST"; case $ST in
setup)
  mkdir -p $CH $N/output; [ -e $N/camera_calibration/rectified ] || ln -s $E/camera_calibration/rectified $N/camera_calibration/rectified
  [ -d $CH/$CN ] || cp -al $E/camera_calibration/chunks/$CN $CH/$CN
  rm -f $CH/$CN/sparse/0/test.txt; cp $S/experimental/h3dgs/camera_calibration/chunks/$CN/sparse/0/test.txt $CH/$CN/sparse/0/test.txt   # real file: the 1-in-10 list
  [ -e $N/output/scaffold ] || cp -al $E/output/scaffold $N/output/scaffold; cp $E/export_meta.json $N/export_meta.json 2>/dev/null
  say "setup: $(python3 -c "
import struct; f=open('$CH/$CN/sparse/0/images.bin','rb'); n=struct.unpack('<Q',f.read(8))[0]; nm=[]
for _ in range(n):
    f.read(64); b=b''
    while (c:=f.read(1))!=b'\x00': b+=c
    k=struct.unpack('<Q',f.read(8))[0]; f.read(24*k); nm.append(b.decode())
t={l.strip() for l in open('$CH/$CN/sparse/0/test.txt') if l.strip()}; print(n,'images,', len(t & set(nm)),'held out,', n-len(t & set(nm)),'trained')")" ;;
train) mkdir -p $T; cd $REPO; t0=$(date +%s)
  python -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images -d ../../rectified/depths --scaffold_file $SC --skybox_locked \
    --exposure_lr_init 0.0 --eval -s $CH/$CN --model_path $T --bounds_file $CH/$CN \
    --iterations 60000 --position_lr_max_steps 60000 --densify_until_iter 45000 --densify_grad_threshold 0.0075 --exposure_lr_init 0.001 > $T/train.log 2>&1
  say "train rc=$? in $(( $(date +%s)-t0 ))s; $(ls -d $T/point_cloud/iteration_* 2>/dev/null | tail -1)" ;;
hier) cd $REPO; PLY=$(ls -d $T/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -1)/point_cloud.ply
  submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH/$CN $T $SC > $T/hier.log 2>&1
  [ -e $T/hierarchy.hier ] && say "hierarchy $(du -h $T/hierarchy.hier | cut -f1)" || { say "no hierarchy"; exit 1; } ;;
post) cd $REPO; t0=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
    -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH/$CN --model_path $T --hierarchy $T/hierarchy.hier --exposure_lr_init 0.001 > $T/post.log 2>&1
  [ -e $T/hierarchy.hier_opt ] && say "post-opt in $(( $(date +%s)-t0 ))s" || { say "no hier_opt"; exit 1; } ;;
eval) cd $REPO; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 1800 python /home/paperspace/logs/h3dgs_eval_chunk.py $N --hier output/trained_chunks/$CN/hierarchy.hier_opt --taus 0 --only_chunk $CN --out output/eval_chunk_$CN --save 12 2>&1 | grep -aE "^\[eval\] tau|Error" | cut -c1-260 | tee -a $L ;;
cell) [ -d $NB ] && mv $NB ${NB}_old_$(date +%H%M)
  SVN=05_13D_Jackal PROJ=$N CN=$CN OUT=$NB bash $NAT/native_sam3_cell.sh > /home/paperspace/logs/retrain_05_1_0_cell.log 2>&1
  (cd /home/paperspace/code/nerf_new && pixi run python $NAT/build_text_bank.py --manifest $NB/supervision/trees_only/manifest.json --hierarchy-json $S/prod/bateleur/scene_graph/marker_hierarchy.json --out $NB/text_bank.npz 2>&1 | grep -a "text-bank\|Error" | tee -a $L)
  [ -f $NB/features_B_bg2share.bin ] && say "cell ok" || { say "cell FAILED"; exit 1; } ;;
score) cd /home/paperspace/code && python automation/h3dgs/fruit_densify/fruit_bc_score.py $N $NB recipe01 2>&1 | grep -a "fruit-bc\|Error\|Traceback" | tee -a $L ;;
esac; done
say "=== done: $STAGES"
