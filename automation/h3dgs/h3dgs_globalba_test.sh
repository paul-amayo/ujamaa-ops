#!/bin/bash
# Candidate fix for chunk misregistration: ONE global bundle adjustment of the whole survey (poses free,
# intrinsics fixed) on the fixed-pose triangulation, Sim(3)-snapped back onto the export frame, then
# measure per-camera shift and reprojection error. Chunks would then be built with --skip_bundle_adjustment.
L=/home/paperspace/logs/h3dgs_globalba.log; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
CC=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/camera_calibration
export PATH=/home/paperspace/logs/h3dgs_bin:$PATH
rm -rf $CC/globalba; mkdir -p $CC/globalba/sparse/raw $CC/globalba/sparse/0
t0=$(date +%s)
colmap bundle_adjuster --input_path $CC/aligned/sparse/0 --output_path $CC/globalba/sparse/raw \
  --BundleAdjustment.refine_focal_length 0 --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 \
  --BundleAdjustment.function_tolerance 0.000001 --BundleAdjustment.max_num_iterations 100 --BundleAdjustment.max_linear_solver_iterations 200 \
  ${BA_GPU:+--BundleAdjustment.use_gpu 1} > /home/paperspace/logs/h3dgs_globalba_colmap.log 2>&1
say "global BA rc=$? in $(( $(date +%s)-t0 ))s ($(grep -aE "Final cost|iterations" /home/paperspace/logs/h3dgs_globalba_colmap.log | tail -2 | tr '\n' ' '))"
/home/paperspace/miniconda3/envs/h3dgs/bin/python - $CC << 'PYEOF' 2>&1 | tee -a $L
import sys, numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec, Image, Point3D
CC = sys.argv[1]
cams0, ims0, pts0 = read_model(f"{CC}/aligned/sparse/0", ".bin")
cams1, ims1, pts1 = read_model(f"{CC}/globalba/sparse/raw", ".bin")
def centres(ims): return {k: -qvec2rotmat(v.qvec).T @ v.tvec for k, v in ims.items()}
c0, c1 = centres(ims0), centres(ims1); keys = sorted(set(c0) & set(c1))
X0 = np.array([c0[k] for k in keys]); X1 = np.array([c1[k] for k in keys])
# Sim(3) X1 -> X0 (Umeyama)
m0, m1 = X0.mean(0), X1.mean(0); H = (X1 - m1).T @ (X0 - m0); U, S, Vt = np.linalg.svd(H); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ D @ U.T; s = np.trace(np.diag(S) @ D) / ((X1 - m1) ** 2).sum(); t = m0 - s * R @ m1
X1a = (s * (R @ X1.T)).T + t
d = np.linalg.norm(X1a - X0, axis=1)
err0 = np.mean([p.error for p in pts0.values()]); err1 = np.mean([p.error for p in pts1.values()])
print(f"[globalba] scale {s:.5f}; camera-centre change after Sim(3) snap: median {np.median(d)*100:.1f} cm, p90 {np.percentile(d,90)*100:.1f} cm, max {d.max()*100:.1f} cm; reproj error {err0:.2f} -> {err1:.2f} px; points {len(pts0)} -> {len(pts1)}")
# write the snapped model as globalba/sparse/0
ims2 = {}
for k, im in ims1.items():
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    c2w[:3, :3] = R @ c2w[:3, :3]; c2w[:3, 3] = s * (R @ c2w[:3, 3]) + t; w2c = np.linalg.inv(c2w)
    ims2[k] = Image(id=im.id, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=im.camera_id, name=im.name, xys=im.xys, point3D_ids=im.point3D_ids)
pts2 = {k: Point3D(id=p.id, xyz=s * (R @ p.xyz) + t, rgb=p.rgb, error=p.error, image_ids=p.image_ids, point2D_idxs=p.point2D_idxs) for k, p in pts1.items()}
write_model(cams1, ims2, pts2, f"{CC}/globalba/sparse/0", ".bin"); print("[globalba] wrote snapped model globalba/sparse/0")
PYEOF
say "GLOBAL BA TEST DONE"
