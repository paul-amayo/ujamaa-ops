#!/usr/bin/env python
"""Pose-FIXED intrinsics estimate for one H3DGS chunk: extract+match features for the chunk's images (40-NN pairs
from the chunk's poses), triangulate with the frames fixed while COLMAP refines focal length / principal point,
then re-triangulate with (a) the nominal and (b) the refined intrinsics and report the reprojection errors.
  python h3dgs_intrinsics_probe.py <proj> <chunk> [--out <dir>]
"""
import argparse, os, shutil, sqlite3, subprocess, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, Camera, Image, read_cameras_binary, read_points3D_binary
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("chunk"); ap.add_argument("--out", default="")
ap.add_argument("--model", default="PINHOLE", choices=["PINHOLE", "OPENCV"], help="OPENCV adds k1 k2 p1 p2 (initialised at 0) and refines them too")
ap.add_argument("--sparse", default="", help="use this sparse model (cameras/images.bin) instead of the chunk's, e.g. camera_calibration/prior/sparse/0 = the whole survey on its LIO poses")
a = ap.parse_args(); P = Path(a.proj); CH = P / "camera_calibration/chunks" / a.chunk; IM = P / "camera_calibration/rectified/images"
W = Path(a.out) if a.out else P / "camera_calibration" / f"intrinsics_probe_{a.chunk}"; W.mkdir(parents=True, exist_ok=True)
SPARSE = Path(a.sparse) if a.sparse else CH / "sparse/0"
COLMAP = "/home/paperspace/logs/h3dgs_bin/colmap"; REPO = "/home/paperspace/code/hierarchical-3d-gaussians"
def run(cmd, log):
    with open(W / log, "a") as f: r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode: sys.exit(f"{cmd[1]} failed (rc={r.returncode}), see {W/log}")
cams, ims, _ = read_model(str(SPARSE), ".bin"); cam = cams[1]; nom = cam.params.copy()
print(f"[probe] chunk {a.chunk}: {len(ims)} images, nominal {cam.model} {np.round(nom, 2).tolist()}", flush=True)
# 1. image subset (hardlinks) + features with the nominal intrinsics
sub = W / "images"; sub.mkdir(exist_ok=True)
for im in ims.values():
    if not (sub / im.name).exists(): os.link(IM / im.name, sub / im.name)
DB = W / "database.db"
if not DB.exists():
    run([COLMAP, "feature_extractor", "--database_path", str(DB), "--image_path", str(sub), "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
         "--ImageReader.camera_params", ",".join(str(float(x)) for x in nom), "--FeatureExtraction.use_gpu", "1"], "colmap.log")
    # prior model with this database's image ids, for the distance matcher
    name2id = dict((n, i) for i, n in sqlite3.connect(DB).execute("SELECT image_id, name FROM images"))
    prior = {name2id[im.name]: Image(id=name2id[im.name], qvec=im.qvec, tvec=im.tvec, camera_id=1, name=im.name, xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, int)) for im in ims.values()}
    pd = W / "prior/sparse/0"; pd.mkdir(parents=True, exist_ok=True); write_model({1: cam}, prior, {}, str(pd), ".bin")
    run([sys.executable, f"{REPO}/preprocess/make_colmap_custom_matcher_distance.py", "--base_dir", str(pd), "--n_neighbours", "40"], "colmap.log")
    run([COLMAP, "matches_importer", "--database_path", str(DB), "--match_list_path", str(pd / "matching_40.txt"), "--FeatureMatching.use_gpu", "1"], "colmap.log")
    print("[probe] features + 40-NN matches done", flush=True)
pd = W / "prior/sparse/0"
MODEL = a.model
if MODEL == "OPENCV" and cam.model == "PINHOLE": nom = np.concatenate([nom, np.zeros(4)])   # fx fy cx cy k1 k2 p1 p2
def triangulate(tag, params, refine):
    tag = f"{tag}_{MODEL.lower()}"; src = W / f"in_{tag}/sparse/0"; src.mkdir(parents=True, exist_ok=True)
    c2 = Camera(id=1, model=MODEL, width=cam.width, height=cam.height, params=np.array(params, float))
    _, prior, _ = read_model(str(pd), ".bin"); write_model({1: c2}, prior, {}, str(src), ".bin")
    out = W / f"out_{tag}"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    cmd = [COLMAP, "point_triangulator", "--database_path", str(DB), "--image_path", str(sub), "--input_path", str(src), "--output_path", str(out),
           "--Mapper.fix_existing_frames", "1", "--Mapper.ba_refine_focal_length", "1" if refine else "0", "--Mapper.ba_refine_principal_point", "1" if refine else "0",
           "--Mapper.ba_refine_extra_params", "1" if (refine and MODEL == "OPENCV") else "0", "--Mapper.ba_global_max_num_iterations", "50"]
    run(cmd, f"tri_{tag}.log")
    c = read_cameras_binary(str(out / "cameras.bin"))[1]; pts = read_points3D_binary(str(out / "points3D.bin"))
    err = np.array([p.error for p in pts.values()]); tl = np.array([len(p.image_ids) for p in pts.values()])
    print(f"[probe] {tag:14s} intrinsics {np.round(c.params, 2).tolist()}: {len(pts)} points, mean reproj {err.mean():.3f} px, median {np.median(err):.3f}, mean track {tl.mean():.2f}", flush=True)
    return c.params, err.mean()
ref_params, _ = triangulate("refine_posefix", nom, True)
print(f"[probe] pose-fixed refined intrinsics: {np.round(ref_params, 2).tolist()} = change {np.round(100 * (ref_params / nom - 1), 2).tolist()} % vs nominal", flush=True)
_, e_nom = triangulate("nominal", nom, False)
_, e_ref = triangulate("refined", ref_params, False)
print(f"[probe] RESULT: fixed-pose triangulation reprojection error nominal {e_nom:.3f} px -> refined intrinsics {e_ref:.3f} px", flush=True)
