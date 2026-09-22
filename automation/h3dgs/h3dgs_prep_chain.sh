#!/bin/bash
# H3DGS on 05_13D — preprocessing chain (no training):
#  1. blue-frame rates on the WARM demo service (35 blocks resident) for the record
#  2. shrink the demo render service so the GPU has headroom for the H3DGS work
#  3. monocular inverse-depth maps (DA-V2 metric outdoor, cached)
#  4. global SfM with FIXED poses: features + distance-matched pairs + point_triangulator -> aligned/sparse/0
#  5. chunks (make_chunk 30 m cells, visibility-selected cameras) + per-chunk triangulation/BA (prepare_chunk)
#  6. per-chunk depth scales + test.txt into every chunk
L=/home/paperspace/logs/h3dgs_prep.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
set -o pipefail
REPO=/home/paperspace/code/hierarchical-3d-gaussians
PROJ=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs
CC=$PROJ/camera_calibration
IMGS=$CC/rectified/images
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH
PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
cd $REPO

if [ "${START_AT:-1}" -le 2 ]; then
say "1. blue rates on the warm service (hero stretch s=50%)"
kill 1773915 2>/dev/null; sleep 1; kill 1776757 2>/dev/null
for P in drive walk; do S0=0.5 PACE=$P timeout 150 /home/paperspace/envs/hfeval_ft/bin/python /home/paperspace/logs/blue_rate.py 2>&1 | grep -aE "\[blue" | tee -a $L; done

say "2. shrink demo render service to budget 8 GiB / preload 8"
RENDER_VRAM_BUDGET=8 RENDER_PRELOAD=8 bash /home/paperspace/logs/restart_render_glref.sh > /home/paperspace/logs/restart_small.log 2>&1
say "   $(tail -1 /home/paperspace/logs/restart_small.log | cut -c1-160)"
fi

if [ "${START_AT:-1}" -le 3 ]; then
say "3. depth maps"
/home/paperspace/envs/hfeval_ft/bin/python /home/paperspace/logs/h3dgs_depth_05.py > /home/paperspace/logs/h3dgs_depth.log 2>&1 || { say "DEPTH FAILED"; exit 1; }
say "   $(tail -1 /home/paperspace/logs/h3dgs_depth.log)"
fi

if [ "${START_AT:-1}" -le 4 ]; then
say "4. global SfM (fixed poses)"
DB=$CC/rectified/database.db; rm -f $DB
PARAMS=$($PY -c "import json;c=json.load(open('$PROJ/export_meta.json'))['camera'];print(f\"{c['fx']},{c['fy']},{c['cx']},{c['cy']}\")")
t0=$(date +%s)
colmap feature_extractor --database_path $DB --image_path $IMGS --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 1 > /home/paperspace/logs/h3dgs_sfm.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
say "   features in $(( $(date +%s)-t0 ))s"
# image ids in the prior model must match the ids colmap assigned in the database
$PY - << 'EOF' || { echo "ID SYNC FAILED" | tee -a /home/paperspace/logs/h3dgs_prep.log; exit 1; }
import sqlite3, sys, numpy as np, shutil
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, Image
CC = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/camera_calibration")
cams, ims, _ = read_model(str(CC/"poses/sparse/0"), ".bin")
db = sqlite3.connect(str(CC/"rectified/database.db"))
name2id = dict((n, i) for i, n in db.execute("SELECT image_id, name FROM images"))
cam_ids = [r[0] for r in db.execute("SELECT camera_id FROM cameras")]
assert len(cam_ids) == 1, cam_ids
byname = {im.name: im for im in ims.values()}
assert set(byname) == set(name2id), (len(byname), len(name2id))
out = {name2id[n]: Image(id=name2id[n], qvec=im.qvec, tvec=im.tvec, camera_id=cam_ids[0], name=n, xys=im.xys, point3D_ids=im.point3D_ids) for n, im in byname.items()}
cams = {cam_ids[0]: cams[1]._replace(id=cam_ids[0])}
d = CC/"prior/sparse/0"; d.mkdir(parents=True, exist_ok=True)
write_model(cams, out, {}, str(d), ".bin"); shutil.copy(CC/"poses/sparse/0/test.txt", d/"test.txt")
print("ids synced:", len(out), "camera id", cam_ids[0])
EOF
$PY preprocess/make_colmap_custom_matcher_distance.py --base_dir $CC/prior/sparse/0 --n_neighbours 40 || { say "MATCH LIST FAILED"; exit 1; }
say "   pairs: $(wc -l < $CC/prior/sparse/0/matching_40.txt)"
t0=$(date +%s)
colmap matches_importer --database_path $DB --match_list_path $CC/prior/sparse/0/matching_40.txt --FeatureMatching.use_gpu 1 >> /home/paperspace/logs/h3dgs_sfm.log 2>&1 || { say "MATCHING FAILED"; exit 1; }
say "   matching in $(( $(date +%s)-t0 ))s"
t0=$(date +%s)
TRI_EXTRA="--Mapper.fix_existing_frames 1"
mkdir -p $CC/rectified/sparse/0
colmap point_triangulator --database_path $DB --image_path $IMGS --input_path $CC/prior/sparse/0 --output_path $CC/rectified/sparse/0 \
  --Mapper.ba_global_function_tolerance 0.000001 --Mapper.ba_global_max_num_iterations 30 --Mapper.ba_global_max_refinements 3 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 $TRI_EXTRA >> /home/paperspace/logs/h3dgs_sfm.log 2>&1 || { say "TRIANGULATION FAILED"; exit 1; }
say "   triangulation in $(( $(date +%s)-t0 ))s"
$PY - << 'EOF' 2>&1 | tee -a /home/paperspace/logs/h3dgs_prep.log
import sys, numpy as np, shutil
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, qvec2rotmat
CC = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/camera_calibration")
_, ims0, _ = read_model(str(CC/"prior/sparse/0"), ".bin")
cams, ims, pts = read_model(str(CC/"rectified/sparse/0"), ".bin")
c0 = {k: -qvec2rotmat(v.qvec).T @ v.tvec for k, v in ims0.items()}
c1 = {k: -qvec2rotmat(v.qvec).T @ v.tvec for k, v in ims.items()}
d = np.array([np.linalg.norm(c1[k] - c0[k]) for k in ims])
tl = np.array([len(p.image_ids) for p in pts.values()]); err = np.array([p.error for p in pts.values()])
print(f"[sfm] images {len(ims)}/{len(ims0)} points {len(pts)} track-len mean {tl.mean():.1f} reproj-err mean {err.mean():.2f}px | camera-centre shift vs prior: mean {d.mean()*100:.2f} cm max {d.max()*100:.1f} cm")
a = CC/"aligned/sparse/0"; a.mkdir(parents=True, exist_ok=True)
for f in ("cameras.bin", "images.bin", "points3D.bin"): shutil.copy(CC/"rectified/sparse/0"/f, a/f)
shutil.copy(CC/"prior/sparse/0/test.txt", a/"test.txt")
EOF
fi

say "5. chunks (30 m cells)"
rm -rf $CC/raw_chunks $CC/chunks
$PY preprocess/make_chunk.py --base_dir $CC/aligned/sparse/0 --images_dir $IMGS --chunk_size 30 --lapla_thresh 0 --min_n_cams 50 --max_n_cams 1500 \
  --output_path $CC/raw_chunks 2>&1 | tee -a $L | tail -20
for RC in $(ls $CC/raw_chunks); do
  t0=$(date +%s)
  $PY preprocess/prepare_chunk.py --raw_chunk $CC/raw_chunks/$RC --out_chunk $CC/chunks/$RC --images_dir $IMGS > /home/paperspace/logs/h3dgs_chunk_$RC.log 2>&1 \
    && say "   chunk $RC refined in $(( $(date +%s)-t0 ))s ($(ls $CC/raw_chunks/$RC/bundle_adjustment/images 2>/dev/null | wc -l) images)" \
    || say "   chunk $RC FAILED (see h3dgs_chunk_$RC.log)"
done

say "6. depth scales + test.txt per chunk"
$PY preprocess/make_chunks_depth_scale.py --chunks_dir $CC/chunks --depths_dir $CC/rectified/depths >> $L 2>&1 || say "DEPTH SCALE FAILED"
$PY preprocess/copy_file_to_chunks.py --file_path $CC/aligned/sparse/0/test.txt --chunks_path $CC/chunks >> $L 2>&1
for c in $(ls $CC/chunks); do say "   $c: $(ls $CC/chunks/$c/sparse/0 | tr '\n' ' ') center=$(cat $CC/chunks/$c/center.txt) extent=$(cat $CC/chunks/$c/extent.txt)"; done
if [ "$(ls -d $CC/chunks/*/sparse/0 2>/dev/null | wc -l)" -gt 0 ]; then say "PREP DONE"; else say "PREP FAILED (no chunks)"; fi
