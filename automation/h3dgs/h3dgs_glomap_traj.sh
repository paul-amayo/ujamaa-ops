#!/bin/bash
# Improve a survey's trajectory with a from-scratch GLOBAL SfM (GPU GLOMAP) over all keyframes, pairs restricted to
# the 40 nearest cameras of the LIO prior (no cross-tunnel false loops), then a robust Sim(3) onto the LIO frame.
# Writes <proj>/camera_calibration/glomap_aligned/sparse/0 (+ test.txt) and prints how far the cameras moved vs the
# prior and vs the global-BA poses. Does NOT touch the project's aligned model or chunks.
#   usage: h3dgs_glomap_traj.sh <proj_dir>
PROJ=${1:?proj}; CC=$PROJ/camera_calibration; IMGS=$CC/rectified/images
L=/home/paperspace/logs/h3dgs_glomap_traj.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
REPO=/home/paperspace/code/hierarchical-3d-gaussians; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
export PATH=/home/paperspace/logs/h3dgs_bin:$PATH
export LD_LIBRARY_PATH=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
cd $REPO; W=$CC/glomap; mkdir -p $W; DB=$W/database.db
say "=== GLOMAP trajectory for $(basename $(dirname $(dirname $PROJ)))"
if [ ! -e $DB ]; then
  t0=$(date +%s)
  PARAMS=$($PY -c "import json;c=json.load(open('$PROJ/export_meta.json'))['camera'];print(f\"{c['fx']},{c['fy']},{c['cx']},{c['cy']}\")")
  colmap feature_extractor --database_path $DB --image_path $IMGS --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 1 > $W/colmap.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
  # pair list from the LIO prior (40 nearest cameras), image ids remapped to this database
  $PY - $CC $DB << 'EOF' || { say "PAIR LIST FAILED"; exit 1; }
import sys, sqlite3, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, Image
CC, DB = Path(sys.argv[1]), sys.argv[2]
cams, ims, _ = read_model(str(CC/"poses/sparse/0"), ".bin")
name2id = dict((n, i) for i, n in sqlite3.connect(DB).execute("SELECT image_id, name FROM images"))
out = {name2id[im.name]: Image(id=name2id[im.name], qvec=im.qvec, tvec=im.tvec, camera_id=1, name=im.name, xys=im.xys, point3D_ids=im.point3D_ids) for im in ims.values()}
d = CC/"glomap/prior/sparse/0"; d.mkdir(parents=True, exist_ok=True); write_model({1: cams[1]}, out, {}, str(d), ".bin")
EOF
  $PY preprocess/make_colmap_custom_matcher_distance.py --base_dir $CC/glomap/prior/sparse/0 --n_neighbours 40 >> $W/colmap.log 2>&1
  colmap matches_importer --database_path $DB --match_list_path $CC/glomap/prior/sparse/0/matching_40.txt --FeatureMatching.use_gpu 1 >> $W/colmap.log 2>&1 || { say "MATCHING FAILED"; exit 1; }
  say "features + $(wc -l < $CC/glomap/prior/sparse/0/matching_40.txt) LIO-neighbour pairs matched in $(( $(date +%s)-t0 ))s"
fi
t0=$(date +%s); rm -rf $W/sparse; mkdir -p $W/sparse
$GB mapper --database_path $DB --image_path $IMGS --output_path $W/sparse > $W/glomap.log 2>&1
say "glomap mapper rc=$? in $(( $(date +%s)-t0 ))s: $(grep -aE "Reconstruction|registered|images" $W/glomap.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
$PY - $CC << 'EOF' 2>&1 | tee -a $L
import sys, glob, numpy as np, shutil
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec, Image, Point3D
CC = Path(sys.argv[1])
recs = sorted(glob.glob(str(CC/"glomap/sparse/*/images.bin")))
best = max(recs, key=lambda r: len(read_model(str(Path(r).parent), ".bin")[1])); rdir = Path(best).parent
cams, ims, pts = read_model(str(rdir), ".bin")
_, prior, _ = read_model(str(CC/"prior/sparse/0"), ".bin"); _, ba, _ = read_model(str(CC/"aligned/sparse/0"), ".bin")
cen = lambda I: {v.name: -qvec2rotmat(v.qvec).T @ v.tvec for v in I.values()}
cg, cp, cb = cen(ims), cen(prior), cen(ba)
keys = sorted(set(cg) & set(cp)); X1 = np.array([cg[k] for k in keys]); X0 = np.array([cp[k] for k in keys])
def umeyama(X1, X0):
    m0, m1 = X0.mean(0), X1.mean(0); U, S, Vt = np.linalg.svd((X1 - m1).T @ (X0 - m0)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; s = np.trace(np.diag(S) @ D) / ((X1 - m1) ** 2).sum(); return s, R, m0 - s * R @ m1
sel = np.ones(len(keys), bool)
for it in range(5):   # trimmed Sim(3): drop the worst 10 % and refit
    s, R, t = umeyama(X1[sel], X0[sel]); res = np.linalg.norm((s * (R @ X1.T)).T + t - X0, axis=1); sel = res <= np.percentile(res, 90)
res_all = np.linalg.norm((s * (R @ X1.T)).T + t - X0, axis=1)
print(f"[glomap] {rdir.name}: registered {len(ims)}/{len(prior)} images, {len(pts)} points, reproj {np.mean([p.error for p in pts.values()]):.2f} px; Sim(3) to LIO: scale {s:.4f}, residual vs LIO prior median {np.median(res_all)*100:.1f} cm p90 {np.percentile(res_all,90)*100:.1f} cm max {res_all.max()*100:.1f} cm")
Xb = np.array([cb[k] for k in keys]); Xg = (s * (R @ X1.T)).T + t; d = np.linalg.norm(Xg - Xb, axis=1)
print(f"[glomap] vs the global-BA poses: median {np.median(d)*100:.1f} cm p90 {np.percentile(d,90)*100:.1f} cm; missing images: {len(prior)-len(ims)}")
out = {}
for k, im in ims.items():
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    c2w[:3, :3] = R @ c2w[:3, :3]; c2w[:3, 3] = s * (R @ c2w[:3, 3]) + t; w2c = np.linalg.inv(c2w)
    out[k] = Image(id=k, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=im.camera_id, name=im.name, xys=im.xys, point3D_ids=im.point3D_ids)
pts2 = {k: Point3D(id=p.id, xyz=s * (R @ p.xyz) + t, rgb=p.rgb, error=p.error, image_ids=p.image_ids, point2D_idxs=p.point2D_idxs) for k, p in pts.items()}
d = CC/"glomap_aligned/sparse/0"; d.mkdir(parents=True, exist_ok=True); write_model(cams, out, pts2, str(d), ".bin"); shutil.copy(CC/"prior/sparse/0/test.txt", d/"test.txt")
print(f"[glomap] wrote {d}")
EOF
say "GLOMAP TRAJECTORY DONE"
