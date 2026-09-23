#!/bin/bash
# Paul (2026-09-23 22:2x): "sequential bundle adjustment with no poses" — whole-survey SfM with SEQUENTIAL pairs only
# (frame i vs i±overlap in trajectory order: no cross-row pairs by construction, no odometry prior, camera fixed),
# solved (a) by GLOMAP and (b) by COLMAP's incremental mapper; each result is Sim(3)-aligned onto the LiDAR
# trajectory and its residual/drift reported.
#   usage: h3dgs_sequential_sfm.sh <proj_dir> [overlap=20]
P=${1:?proj}; OV=${2:-20}; CC=$P/camera_calibration; IM=$CC/rectified/images; W=$CC/sequential; mkdir -p $W
L=/home/paperspace/logs/h3dgs_sequential_sfm.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
GB=/home/paperspace/code/glomap/build_gpu/glomap/glomap
export LD_LIBRARY_PATH=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
PARAMS=$($PY -c "import json;c=json.load(open('$P/export_meta.json'))['camera'];print(f\"{c['fx']},{c['fy']},{c['cx']},{c['cy']}\")")
say "=== sequential SfM on $(basename $(dirname $(dirname $P))) ($(ls $IM | wc -l) frames, overlap $OV, camera fixed $PARAMS)"
if [ ! -e $W/database.db ]; then
  t0=$(date +%s)
  colmap feature_extractor --database_path $W/database.db --image_path $IM --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 1 > $W/colmap.log 2>&1 || { say "FEATURES FAILED"; exit 1; }
  colmap sequential_matcher --database_path $W/database.db --SequentialMatching.overlap $OV --SequentialMatching.loop_detection 0 --FeatureMatching.use_gpu 1 >> $W/colmap.log 2>&1 || { say "SEQUENTIAL MATCHING FAILED"; exit 1; }
  say "features + sequential matches (overlap $OV) in $(( $(date +%s)-t0 ))s"
fi
report() {   # <tag> <model dir>
  $PY - $CC $2 $1 << 'EOF' | tee -a $L
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, qvec2rotmat
CC, M, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cams, ims, pts = read_model(str(M), ".bin"); _, prior, _ = read_model(str(CC / "prior/sparse/0"), ".bin")
cen = lambda I: {v.name: -qvec2rotmat(v.qvec).T @ v.tvec for v in I.values()}; cg, cp = cen(ims), cen(prior)
keys = sorted(set(cg) & set(cp)); X1 = np.array([cg[k] for k in keys]); X0 = np.array([cp[k] for k in keys])
def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; s = np.trace(np.diag(S) @ D) / ((A - ma) ** 2).sum(); return s, R, mb - s * R @ ma
s, R, t = umeyama(X1, X0); res = np.linalg.norm((s * (R @ X1.T)).T + t - X0, axis=1)
err = np.array([p.error for p in pts.values()]) if pts else np.array([np.nan])
print(f"[seq] {tag}: registered {len(ims)}/{len(prior)}, {len(pts)} points, reproj {err.mean():.2f} px | ONE Sim(3) onto the LiDAR trajectory: residual median {np.median(res)*100:.1f} cm p90 {np.percentile(res,90)*100:.1f} cm max {res.max()*100:.1f} cm (scale {s:.4f})", flush=True)
# drift profile: residual per 200-keyframe segment along the survey
order = sorted(keys); idx = {k: i for i, k in enumerate(order)}; r_ord = np.array([res[keys.index(k)] for k in order])
print("[seq]   residual per 200-kf segment (cm):", " ".join(f"{np.median(r_ord[i:i+200])*100:.0f}" for i in range(0, len(order), 200)), flush=True)
EOF
}
t0=$(date +%s); rm -rf $W/glomap; mkdir -p $W/glomap
$GB mapper --database_path $W/database.db --image_path $IM --output_path $W/glomap --BundleAdjustment.optimize_intrinsics 0 --BundleAdjustment.optimize_principal_point 0 > $W/glomap.log 2>&1
say "GLOMAP (sequential pairs) rc=$? in $(( $(date +%s)-t0 ))s"; best=$(for d in $W/glomap/*/; do echo "$(stat -c %s $d/images.bin) $d"; done | sort -n | tail -1 | cut -d' ' -f2); [ -n "$best" ] && report glomap_sequential $best
t0=$(date +%s); rm -rf $W/colmap; mkdir -p $W/colmap
colmap mapper --database_path $W/database.db --image_path $IM --output_path $W/colmap --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.ba_global_use_gpu 1 > $W/colmap_mapper.log 2>&1
say "COLMAP incremental mapper rc=$? in $(( $(date +%s)-t0 ))s ($(ls $W/colmap | wc -l) models)"; best=$(for d in $W/colmap/*/; do echo "$(stat -c %s $d/images.bin) $d"; done | sort -n | tail -1 | cut -d' ' -f2); [ -n "$best" ] && report colmap_sequential $best
say "SEQUENTIAL SFM DONE"
