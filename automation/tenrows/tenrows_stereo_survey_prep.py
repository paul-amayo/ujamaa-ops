"""Survey-wide stereo project for ten_rows (2026-09-26): rgb_zed's export (left keyframes, ZED-conf camera, LiDAR seed,
scaffold) plus the ZED RIGHT frame of every keyframe, placed with the rigid rig transform measured on chunk 1_1
(0.1179-unit x-offset, identity rotation; stereo_reg_11/stereo_model/rig.json). Held-out timestamps are excluded on
both cameras. Output: experimental/h3dgs_rgb_zed_stereo ready for h3dgs_survey.sh (export step skipped: export_meta.json
copied; depths for the left frames and the rgb_zed scaffold hardlinked; the survey script computes the right depths,
runs SfM with the given poses fixed, global BA, chunks, LiDAR seed, chunk training, merge and eval).
Run with nerf_new python3.10 + PYTHONPATH of the cp310 aru_py_logger build (right-frame extraction inside)."""
import json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, write_model, Image, qvec2rotmat, rotmat2qvec
R_ = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); P = R_ / "experimental/h3dgs_rgb_zed"; X = R_ / "experimental/h3dgs_rgb_zed_stereo"; RK = R_ / "experimental/stereo_right_kf"
rig = np.array(json.load(open(R_ / "experimental/stereo_reg_11/stereo_model/rig.json"))["T_rig_left_to_right"])
cams, ims, _ = read_model(str(P / "camera_calibration/poses/sparse/0"), ".bin"); names = sorted(im.name for im in ims.values())
# 1. right frames for every keyframe (the extractor matches by left timestamp; existing files are rewritten, 3 min)
nf = Path("/home/paperspace/logs/tenrows_all_kf_names.txt"); nf.write_text("\n".join(names) + "\n")
if sum(1 for n in names if (RK / (n[:-4] + "_R.png")).exists()) < len(names):
    subprocess.run([sys.executable, "/home/paperspace/logs/tenrows_right_kf_extract.py", "--names", str(nf), "--out", str(RK)], check=True)
have = [n for n in names if (RK / (n[:-4] + "_R.png")).exists()]; print(f"[prep] right frames available for {len(have)}/{len(names)} keyframes", flush=True)
# 2. project layout
CC = X / "camera_calibration"; (CC / "rectified/images").mkdir(parents=True, exist_ok=True); (CC / "rectified/depths").mkdir(parents=True, exist_ok=True); (CC / "poses/sparse/0").mkdir(parents=True, exist_ok=True)
SC = X / "output/scaffold/point_cloud/iteration_30000"; SC.mkdir(parents=True, exist_ok=True)
def link(src, dst):
    if dst.exists(): dst.unlink()
    os.link(src, dst)
for n in names:
    link(P / "camera_calibration/rectified/images" / n, CC / "rectified/images" / n); link(P / "camera_calibration/rectified/depths" / n, CC / "rectified/depths" / n)
    r = n[:-4] + "_R.png"
    if (RK / r).exists(): link(RK / r, CC / "rectified/images" / r)
for f in ("point_cloud.ply", "pc_info.txt"): link(P / "output/scaffold/point_cloud/iteration_30000" / f, SC / f)
shutil.copy2(P / "export_meta.json", X / "export_meta.json")
# 3. poses: left as exported + right = rig @ left (camera-frame transform: frame-independent)
def w2c_of(im):
    T = np.eye(4); T[:3, :3] = qvec2rotmat(im.qvec); T[:3, 3] = im.tvec; return T
out = dict(ims); nid = max(ims) + 1; cid = list(cams)[0]; nr = 0
for im in sorted(ims.values(), key=lambda i: i.name):
    r = im.name[:-4] + "_R.png"
    if not (RK / r).exists(): continue
    T = rig @ w2c_of(im); out[nid] = Image(id=nid, qvec=rotmat2qvec(T[:3, :3]), tvec=T[:3, 3], camera_id=cid, name=r, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64)); nid += 1; nr += 1
write_model(cams, out, {}, str(CC / "poses/sparse/0"), ".bin")
test = [l.strip() for l in open(P / "camera_calibration/poses/sparse/0/test.txt") if l.strip()]
(CC / "poses/sparse/0/test.txt").write_text("\n".join(test + [t[:-4] + "_R.png" for t in test if (RK / (t[:-4] + "_R.png")).exists()]) + "\n")
print(f"[prep] {X}: {len(ims)} left + {nr} right poses, test.txt {len(test)} left + right pairs; images dir {len(list((CC / 'rectified/images').glob('*.png')))} files, depths {len(list((CC / 'rectified/depths').glob('*.png')))} (right depths computed by the survey script); scaffold hardlinked", flush=True)
