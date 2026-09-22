#!/bin/bash
# Hierarchical-3DGS for ONE UJAMAA survey, end to end and idempotent (every stage checks its outputs):
#   export -> mono depth -> fixed-pose global triangulation -> 30 m chunks + per-chunk BA -> depth scales
#   -> coarse scaffold -> per-chunk train/hierarchy/post-opt (smallest chunks first; an OOM is logged and skipped)
#   -> merge of the completed chunks -> held-out render (tau 0/3/6) -> per-chunk held-out PSNR summary.
#   usage: h3dgs_survey.sh <survey_root> <proj_dir>      e.g. .../klapmuts/apr_2026_zed  .../klapmuts/apr_2026_zed/experimental/h3dgs
SURVEY=${1:?survey root}; PROJ=${2:?project dir}
SV=$(basename $SURVEY); L=/home/paperspace/logs/h3dgs_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians; CC=$PROJ/camera_calibration; IMGS=$CC/rectified/images
OUT=$PROJ/output; CH=$CC/chunks; SC=$OUT/scaffold/point_cloud/iteration_30000; FT=/home/paperspace/logs/h3dgs_${SV}_train.log
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python; HF=/home/paperspace/envs/hfeval_ft/bin/python
cd $REPO; mkdir -p $PROJ; T_ALL=$(date +%s)
say "=== $SV -> $PROJ (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used at start)"

# 1. export
if [ ! -e $PROJ/export_meta.json ]; then
  t0=$(date +%s); $PY /home/paperspace/logs/h3dgs_export.py $SURVEY $PROJ >> $L 2>&1 || { say "EXPORT FAILED"; exit 1; }
  say "export in $(( $(date +%s)-t0 ))s: $($PY -c "import json;m=json.load(open('$PROJ/export_meta.json'));print(m['n_images'],'images',m['n_test'],'test',m['convention'])")"
fi
NIMG=$(ls $IMGS | wc -l)
# 2. depth maps
if [ "$(ls $CC/rectified/depths 2>/dev/null | wc -l)" -lt "$NIMG" ]; then
  t0=$(date +%s); H3DGS_PROJ=$PROJ $HF /home/paperspace/logs/h3dgs_depth_05.py > /home/paperspace/logs/h3dgs_${SV}_depth.log 2>&1 || { say "DEPTH FAILED"; exit 1; }
  say "depth maps in $(( $(date +%s)-t0 ))s ($(ls $CC/rectified/depths | wc -l))"
fi
# 3. global SfM with fixed poses
if [ ! -e $CC/aligned/sparse/0/points3D.bin ]; then
  DB=$CC/rectified/database.db; rm -f $DB; t0=$(date +%s)
  PARAMS=$($PY -c "import json;c=json.load(open('$PROJ/export_meta.json'))['camera'];print(f\"{c['fx']},{c['fy']},{c['cx']},{c['cy']}\")")
  colmap feature_extractor --database_path $DB --image_path $IMGS --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 1 > /home/paperspace/logs/h3dgs_${SV}_sfm.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
  $PY - $CC << 'EOF' || { say "ID SYNC FAILED"; exit 1; }
import sqlite3, sys, shutil
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, Image
CC = Path(sys.argv[1]); cams, ims, _ = read_model(str(CC/"poses/sparse/0"), ".bin")
db = sqlite3.connect(str(CC/"rectified/database.db")); name2id = dict((n, i) for i, n in db.execute("SELECT image_id, name FROM images"))
cam_ids = [r[0] for r in db.execute("SELECT camera_id FROM cameras")]; assert len(cam_ids) == 1, cam_ids
byname = {im.name: im for im in ims.values()}; assert set(byname) == set(name2id), (len(byname), len(name2id))
out = {name2id[n]: Image(id=name2id[n], qvec=im.qvec, tvec=im.tvec, camera_id=cam_ids[0], name=n, xys=im.xys, point3D_ids=im.point3D_ids) for n, im in byname.items()}
d = CC/"prior/sparse/0"; d.mkdir(parents=True, exist_ok=True)
write_model({cam_ids[0]: cams[1]._replace(id=cam_ids[0])}, out, {}, str(d), ".bin"); shutil.copy(CC/"poses/sparse/0/test.txt", d/"test.txt")
EOF
  $PY preprocess/make_colmap_custom_matcher_distance.py --base_dir $CC/prior/sparse/0 --n_neighbours 40 >> $L 2>&1 || { say "MATCH LIST FAILED"; exit 1; }
  colmap matches_importer --database_path $DB --match_list_path $CC/prior/sparse/0/matching_40.txt --FeatureMatching.use_gpu 1 >> /home/paperspace/logs/h3dgs_${SV}_sfm.log 2>&1 || { say "MATCHING FAILED"; exit 1; }
  mkdir -p $CC/rectified/sparse/0
  colmap point_triangulator --database_path $DB --image_path $IMGS --input_path $CC/prior/sparse/0 --output_path $CC/rectified/sparse/0 \
    --Mapper.ba_global_function_tolerance 0.000001 --Mapper.ba_global_max_num_iterations 30 --Mapper.ba_global_max_refinements 3 \
    --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.fix_existing_frames 1 >> /home/paperspace/logs/h3dgs_${SV}_sfm.log 2>&1 || { say "TRIANGULATION FAILED"; exit 1; }
  mkdir -p $CC/aligned/sparse/0 && cp $CC/rectified/sparse/0/*.bin $CC/aligned/sparse/0/ && cp $CC/prior/sparse/0/test.txt $CC/aligned/sparse/0/
  STATS=$($PY - $CC << 'EOF'
import sys, numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model
_, ims, pts = read_model(sys.argv[1] + "/aligned/sparse/0", ".bin")
tl = np.array([len(p.image_ids) for p in pts.values()]); err = np.array([p.error for p in pts.values()])
print(f"{len(ims)} images, {len(pts)} points, track {tl.mean():.1f}, reproj {err.mean():.2f} px")
EOF
)
  say "SfM in $(( $(date +%s)-t0 ))s: $STATS"
fi
# 4. chunks + per-chunk BA + depth scales + test split
if [ "$(ls $CH/*/sparse/0/depth_params.json 2>/dev/null | wc -l)" -eq 0 ]; then
  t0=$(date +%s); rm -rf $CC/raw_chunks $CH
  $PY preprocess/make_chunk.py --base_dir $CC/aligned/sparse/0 --images_dir $IMGS --chunk_size 30 --lapla_thresh 0 --min_n_cams 50 --max_n_cams 1500 --output_path $CC/raw_chunks >> $L 2>&1
  for RC in $(ls $CC/raw_chunks); do
    t1=$(date +%s)
    $PY preprocess/prepare_chunk.py --raw_chunk $CC/raw_chunks/$RC --out_chunk $CH/$RC --images_dir $IMGS > /home/paperspace/logs/h3dgs_${SV}_chunk_$RC.log 2>&1 \
      && say "chunk $RC refined in $(( $(date +%s)-t1 ))s" || say "chunk $RC BA FAILED"
  done
  $PY preprocess/make_chunks_depth_scale.py --chunks_dir $CH --depths_dir $CC/rectified/depths >> $L 2>&1 || { say "DEPTH SCALE FAILED"; exit 1; }
  $PY preprocess/copy_file_to_chunks.py --file_path $CC/aligned/sparse/0/test.txt --chunks_path $CH >> $L 2>&1
  say "chunks in $(( $(date +%s)-t0 ))s: $(for c in $(ls $CH); do echo -n "$c=$($PY -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/$c/sparse/0/images.bin')))") "; done)"
fi
# 5. scaffold
mkdir -p $OUT/trained_chunks
if [ ! -e $SC/point_cloud.ply ]; then
  t0=$(date +%s)
  python train_coarse.py -s $CC/aligned --save_iterations -1 -i ../rectified/images --skybox_num 100000 --model_path $OUT/scaffold --exposure_lr_init 0.0 --eval >> $FT 2>&1
  say "scaffold rc=$? in $(( $(date +%s)-t0 ))s $( [ -e $SC/point_cloud.ply ] && echo OK || echo MISSING)"
  [ -e $SC/point_cloud.ply ] || { say "SCAFFOLD FAILED"; exit 1; }
fi
# 6. chunks, smallest first
ORDER=$(for c in $(ls $CH); do echo "$($PY -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/$c/sparse/0/images.bin')))") $c"; done | sort -n | awk '{print $2}')
for c in $ORDER; do
  T=$OUT/trained_chunks/$c; mkdir -p $T
  if [ ! -e $T/point_cloud/iteration_30000/point_cloud.ply ]; then
    t0=$(date +%s)
    python -u train_single.py --save_iterations -1 -i ../../rectified/images -d ../../rectified/depths --scaffold_file $SC --skybox_locked \
      --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --bounds_file $CH/$c >> $FT 2>&1
    say "train chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s"
    [ -e $T/point_cloud/iteration_30000/point_cloud.ply ] || { say "chunk $c: no point cloud — skipped"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier ]; then
    submodules/gaussianhierarchy/build/GaussianHierarchyCreator $T/point_cloud/iteration_30000/point_cloud.ply $CH/$c $T $SC >> $FT 2>&1
    [ -e $T/hierarchy.hier ] || { say "chunk $c: no hierarchy — skipped"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier_opt ]; then
    t0=$(date +%s)
    python -u train_post.py --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
      -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --hierarchy $T/hierarchy.hier >> $FT 2>&1
    say "post-opt chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt (OOM?)')"
  fi
done
DONE=$(for c in $ORDER; do [ -e $OUT/trained_chunks/$c/hierarchy.hier_opt ] && echo -n "$c "; done)
say "chunks complete: $DONE (of $(echo $ORDER | wc -w))"
[ -n "$DONE" ] || { say "NO CHUNK COMPLETED"; exit 1; }
# 7. merge + held-out render
if [ ! -e $OUT/merged.hier ] || [ "$(cat $OUT/merged.chunks 2>/dev/null)" != "$DONE" ]; then
  t0=$(date +%s); rm -f $OUT/merged.hier
  submodules/gaussianhierarchy/build/GaussianHierarchyMerger $OUT/trained_chunks 0 $CH $OUT/merged.hier $DONE >> $FT 2>&1
  say "merge rc=$? in $(( $(date +%s)-t0 ))s ($(du -h $OUT/merged.hier 2>/dev/null | cut -f1))"; echo "$DONE" > $OUT/merged.chunks
  [ -f $OUT/merged.hier ] || { say "NO MERGED HIERARCHY"; exit 1; }
  t0=$(date +%s)
  python render_hierarchy.py -s $CC/aligned -i ../rectified/images --model_path $OUT --hierarchy $OUT/merged.hier --out_dir $OUT/renders --eval --scaffold_file $SC --taus 0 3 6 > /home/paperspace/logs/h3dgs_${SV}_render_eval.log 2>&1
  say "held-out render in $(( $(date +%s)-t0 ))s: $(grep -aE "tau:" /home/paperspace/logs/h3dgs_${SV}_render_eval.log | tr '\n' ' ')"
fi
say "=== $SV DONE in $(( ($(date +%s)-T_ALL)/60 )) min"
