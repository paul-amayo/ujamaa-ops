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
fi
# 3b. ONE global bundle adjustment (poses free, intrinsics fixed), Sim(3)-snapped onto the prior frame, then
#     re-triangulated with poses FIXED -> aligned/sparse/0. Per-chunk BAs disagreed by 6-24 cm on shared cameras
#     (measured on 05 2026-09-22); a single survey-wide solution is consistent by construction.
if [ "${H3DGS_SKIP_GLOBAL_BA:-0}" = 1 ] && [ "$(cat $CC/aligned/RECIPE 2>/dev/null)" != "fixedposes" ]; then
  # A/B variant: keep the exported (e.g. per-block-refined) poses exactly; triangulation only
  echo fixedposes > $CC/aligned/RECIPE; rm -f $CC/rectified/database.db*; say "global BA SKIPPED (H3DGS_SKIP_GLOBAL_BA=1): fixed-pose triangulation kept as aligned/sparse/0"
fi
if [ "${H3DGS_SKIP_GLOBAL_BA:-0}" != 1 ] && [ "$(cat $CC/aligned/RECIPE 2>/dev/null)" != "globalba" ]; then
  DB=$CC/rectified/database.db; t0=$(date +%s); rm -rf $CC/globalba; mkdir -p $CC/globalba/sparse/raw $CC/globalba/sparse/snap
  colmap bundle_adjuster --input_path $CC/rectified/sparse/0 --output_path $CC/globalba/sparse/raw \
    --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 \
    --BundleAdjustment.function_tolerance 0.000001 --BundleAdjustment.max_num_iterations 100 --BundleAdjustment.max_linear_solver_iterations 200 \
    > /home/paperspace/logs/h3dgs_${SV}_globalba.log 2>&1 || { say "GLOBAL BA FAILED"; exit 1; }
  SNAP=$($PY - $CC << 'PYEOF'
import sys, numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec, Image
CC = sys.argv[1]
_, ims0, _ = read_model(f"{CC}/rectified/sparse/0", ".bin"); cams1, ims1, _ = read_model(f"{CC}/globalba/sparse/raw", ".bin")
cen = lambda ims: {k: -qvec2rotmat(v.qvec).T @ v.tvec for k, v in ims.items()}
c0, c1 = cen(ims0), cen(ims1); keys = sorted(set(c0) & set(c1))
X0 = np.array([c0[k] for k in keys]); X1 = np.array([c1[k] for k in keys]); m0, m1 = X0.mean(0), X1.mean(0)
U, S, Vt = np.linalg.svd((X1 - m1).T @ (X0 - m0)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ D @ U.T; s = np.trace(np.diag(S) @ D) / ((X1 - m1) ** 2).sum(); t = m0 - s * R @ m1
d = np.linalg.norm((s * (R @ X1.T)).T + t - X0, axis=1)
out = {}
for k, im in ims1.items():
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    c2w[:3, :3] = R @ c2w[:3, :3]; c2w[:3, 3] = s * (R @ c2w[:3, 3]) + t; w2c = np.linalg.inv(c2w)
    out[k] = Image(id=im.id, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=im.camera_id, name=im.name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), int))
write_model(cams1, out, {}, f"{CC}/globalba/sparse/snap", ".bin")
print(f"scale {s:.5f}, camera shift median {np.median(d)*100:.1f} cm p90 {np.percentile(d,90)*100:.1f} cm max {d.max()*100:.1f} cm")
PYEOF
)
  colmap point_triangulator --database_path $DB --image_path $IMGS --input_path $CC/globalba/sparse/snap --output_path $CC/aligned/sparse/0 \
    --Mapper.ba_global_function_tolerance 0.000001 --Mapper.ba_global_max_num_iterations 30 --Mapper.ba_global_max_refinements 3 \
    --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.fix_existing_frames 1 >> /home/paperspace/logs/h3dgs_${SV}_globalba.log 2>&1 || { say "RE-TRIANGULATION FAILED"; exit 1; }
  cp $CC/prior/sparse/0/test.txt $CC/aligned/sparse/0/; echo globalba > $CC/aligned/RECIPE
  rm -f $CC/rectified/database.db*   # features+matches are dead once the model is triangulated
  say "global BA in $(( $(date +%s)-t0 ))s: $(grep -a 'Final cost' /home/paperspace/logs/h3dgs_${SV}_globalba.log | head -1 | tr -s ' ') | snap: $SNAP"
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
# per-chunk idempotent: a chunk is prepared once (sparse/0/depth_params.json); a recipe change wipes everything
if [ "$(cat $CH/RECIPE 2>/dev/null)" != "$(cat $CC/aligned/RECIPE)" ] && [ -e $CH/RECIPE ]; then say "chunk recipe changed ($(cat $CH/RECIPE) -> $(cat $CC/aligned/RECIPE)): rebuilding chunks"; rm -rf $CC/raw_chunks $CH; fi
NEED=0; [ -e $CC/raw_chunks ] || NEED=1
for RC in $(ls $CC/raw_chunks 2>/dev/null); do if [ -n "${H3DGS_ONLY_CHUNKS:-}" ] && ! [[ " $H3DGS_ONLY_CHUNKS " == *" $RC "* ]]; then continue; fi; [ -e $CH/$RC/sparse/0/depth_params.json ] || NEED=1; done
if [ "$NEED" = 1 ]; then
  t0=$(date +%s)
  [ -e $CC/raw_chunks ] || $PY preprocess/make_chunk.py --base_dir $CC/aligned/sparse/0 --images_dir $IMGS --chunk_size 30 --lapla_thresh 0 --min_n_cams 50 --max_n_cams 1500 --output_path $CC/raw_chunks >> $L 2>&1
  SKIPBA="--skip_bundle_adjustment"; [ "${H3DGS_CHUNK_BA:-0}" = 1 ] && SKIPBA=""   # H3DGS_CHUNK_BA=1: the original per-chunk BA on top of the aligned poses
  for RC in $(ls $CC/raw_chunks); do
    if [ -n "${H3DGS_ONLY_CHUNKS:-}" ] && ! [[ " $H3DGS_ONLY_CHUNKS " == *" $RC "* ]]; then continue; fi   # chunk tests: prepare only the chunks under test
    [ -e $CH/$RC/sparse/0/depth_params.json ] && continue   # already prepared
    t1=$(date +%s)
    $PY preprocess/prepare_chunk.py --raw_chunk $CC/raw_chunks/$RC --out_chunk $CH/$RC --images_dir $IMGS $SKIPBA > /home/paperspace/logs/h3dgs_${SV}_chunk_$RC.log 2>&1 \
      && say "chunk $RC $( [ -z "$SKIPBA" ] && echo "bundle-adjusted" || echo "triangulated (poses fixed)") in $(( $(date +%s)-t1 ))s" || say "chunk $RC TRIANGULATION FAILED"
    rm -rf $CC/raw_chunks/$RC/bundle_adjustment/images $CC/raw_chunks/$RC/bundle_adjustment/stereo $CC/raw_chunks/$RC/bundle_adjustment/database.db*   # transient copies (training reads rectified/images)
  done
  $PY preprocess/make_chunks_depth_scale.py --chunks_dir $CH --depths_dir $CC/rectified/depths >> $L 2>&1 || { say "DEPTH SCALE FAILED"; exit 1; }
  $PY preprocess/copy_file_to_chunks.py --file_path $CC/aligned/sparse/0/test.txt --chunks_path $CH >> $L 2>&1
  cp $CC/aligned/RECIPE $CH/RECIPE; [ -z "$SKIPBA" ] && echo 1 > $CH/CHUNK_BA
  say "chunks in $(( $(date +%s)-t0 ))s: $(for c in $(cd $CH && ls -d */ | tr -d /); do echo -n "$c=$($PY -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/$c/sparse/0/images.bin')))") "; done)"
fi
gpu_wait(){ until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 30; done; }   # concurrent survey instances: never start a trainer into a full GPU
[ "${H3DGS_STOP_AFTER_CHUNKS:-0}" = 1 ] && { say "stopped after the chunk stage (H3DGS_STOP_AFTER_CHUNKS)"; exit 0; }
# 5. scaffold
mkdir -p $OUT/trained_chunks
if [ ! -e $SC/point_cloud.ply ]; then
  t0=$(date +%s)
  gpu_wait 8000; python train_coarse.py --port $((6100 + RANDOM % 900)) -s $CC/aligned --save_iterations -1 -i ../rectified/images --skybox_num 100000 --model_path $OUT/scaffold --exposure_lr_init 0.0 --eval >> $FT 2>&1
  say "scaffold rc=$? in $(( $(date +%s)-t0 ))s $( [ -e $SC/point_cloud.ply ] && echo OK || echo MISSING)"
  [ -e $SC/point_cloud.ply ] || { say "SCAFFOLD FAILED"; exit 1; }
fi
# 6. chunks, smallest first
# H3DGS_ONLY_CHUNKS="1_1 0_0" trains only those chunks (pose/recipe tests); merge+eval then run on whatever is complete
ORDER=$(for c in $(cd $CH && ls -d */ | tr -d /); do echo "$($PY -c "import sys;sys.path.insert(0,'preprocess');from read_write_model import read_images_binary as r;print(len(r('$CH/$c/sparse/0/images.bin')))") $c"; done | sort -n | awk '{print $2}')
for c in $ORDER; do
  if [ -n "${H3DGS_ONLY_CHUNKS:-}" ] && ! [[ " $H3DGS_ONLY_CHUNKS " == *" $c "* ]]; then continue; fi
  T=$OUT/trained_chunks/$c; mkdir -p $T
  # a finished chunk is one with an optimised hierarchy (its ply / raw hierarchy are cleaned up after post-opt)
  if [ -e $T/hierarchy.hier_opt ]; then rm -rf $T/point_cloud $T/hierarchy.hier; continue; fi
  if [ -z "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ]; then
    t0=$(date +%s)
    gpu_wait 12000; python -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images -d ../../rectified/depths --scaffold_file $SC --skybox_locked \
      --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --bounds_file $CH/$c ${H3DGS_TRAIN_EXTRA:-} >> $FT 2>&1   # H3DGS_TRAIN_EXTRA: recipe overrides (densify threshold, iterations, alpha masks ...)
    say "train chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s"
    [ -n "$(ls -d $T/point_cloud/iteration_*/point_cloud.ply 2>/dev/null)" ] || { say "chunk $c: no point cloud — skipped"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier ]; then
    PLY=$(ls -d $T/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -1)/point_cloud.ply   # last saved iteration (H3DGS_TRAIN_EXTRA may lengthen training)
    submodules/gaussianhierarchy/build/GaussianHierarchyCreator $PLY $CH/$c $T $SC >> $FT 2>&1
    [ -e $T/hierarchy.hier ] || { say "chunk $c: no hierarchy — skipped"; continue; }
  fi
  if [ ! -e $T/hierarchy.hier_opt ]; then
    t0=$(date +%s)
    gpu_wait 22000; python -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 \
      -i ../../rectified/images --scaffold_file $SC --exposure_lr_init 0.0 --eval -s $CH/$c --model_path $T --hierarchy $T/hierarchy.hier ${H3DGS_POST_EXTRA:-} >> $FT 2>&1
    say "post-opt chunk $c rc=$? wall=$(( $(date +%s)-t0 ))s $( [ -e $T/hierarchy.hier_opt ] && echo OK || echo 'NO hier_opt (OOM?)')"
  fi
  # keep only what the merge needs: the optimised hierarchy (the ply and the un-optimised hierarchy are its inputs)
  [ -e $T/hierarchy.hier_opt ] && rm -rf $T/point_cloud $T/hierarchy.hier
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
  /home/paperspace/logs/stop_hier_service.sh >/dev/null 2>&1; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/paperspace/logs/h3dgs_eval_compact.py $PROJ --taus 0 3 6 > /home/paperspace/logs/h3dgs_${SV}_eval_compact.log 2>&1
  say "held-out eval (compact) in $(( $(date +%s)-t0 ))s: $(grep -aE "^\[eval\] tau" /home/paperspace/logs/h3dgs_${SV}_eval_compact.log | tr '\n' ' ')"
fi
say "=== $SV DONE in $(( ($(date +%s)-T_ALL)/60 )) min"
