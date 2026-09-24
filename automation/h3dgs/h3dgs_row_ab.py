#!/usr/bin/env python
"""A/B 'more images' on ONE row for H3DGS: build two single-row chunks inside a project from a block's keyframe
transforms (block_NNN_ref) and its full-stream transforms (block_NNN_full_ref) — same refine recipe, same LiDAR seed,
same held-out keyframes (the project's test list restricted to the block) — ready for h3dgs_ablate_chunk.sh.
  python h3dgs_row_ab.py <survey_root> <proj_dir> <block id e.g. 013> [--cell_extent 30]
"""
import argparse, json, os, shutil, sys, numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_cameras_binary, write_model, Image, rotmat2qvec
ap = argparse.ArgumentParser(); ap.add_argument("survey"); ap.add_argument("proj"); ap.add_argument("block"); ap.add_argument("--cell_extent", type=float, default=30.0)
a = ap.parse_args(); S, P = Path(a.survey), Path(a.proj); BL = S / "prod/tassili/blocks_ns/lio_row100"; CC = P / "camera_calibration"
R_W = np.array(json.load(open(P / "export_meta.json"))["world_rotation_to_zup"]); GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
cam = read_cameras_binary(str(CC / "aligned/sparse/0/cameras.bin"))
test_all = set(l.strip() for l in open(CC / "aligned/sparse/0/test.txt") if l.strip())
kf_names = set(Path(f["file_path"]).name for f in json.load(open(BL / f"block_{a.block}/transforms.json"))["frames"])
test = sorted(kf_names & test_all); print(f"[row-ab] block_{a.block}: {len(kf_names)} keyframes, {len(test)} held-out: {test[0]}..{test[-1]}")
IM = CC / "rectified/images"
# LiDAR seed once (block's init, rotated into the project frame)
v = PlyData.read(str(BL / f"block_{a.block}_ref/init_lidar.ply"))["vertex"]; lxyz = (R_W[:3, :3] @ np.stack([v["x"], v["y"], v["z"]], 1).T).T; lrgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
def build(tag, tj_path):
    t = json.load(open(tj_path)); assert t.get("pose_convention", "").startswith("opengl"), tj_path
    ims, centers = {}, []
    for k, f in enumerate(t["frames"]):
        src = Path(f["file_path"]); name = src.name
        if not (IM / name).exists(): os.link(src, IM / name)          # full-stream frames join the project's image folder
        c2w = R_W @ np.array(f["transform_matrix"]) @ GL2CV; w2c = np.linalg.inv(c2w); centers.append(c2w[:3, 3])
        ims[k + 1] = Image(id=k + 1, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), int))
    cd = CC / "chunks" / tag; shutil.rmtree(cd, ignore_errors=True); (cd / "sparse/0").mkdir(parents=True)
    write_model(cam, ims, {}, str(cd / "sparse/0"), ".bin")
    C = np.array(centers); c = C.mean(0); c[2] = 0.0; e = np.array([a.cell_extent, a.cell_extent, 2e12])
    np.savetxt(cd / "center.txt", c[None], fmt="%.6f"); np.savetxt(cd / "extent.txt", e[None], fmt="%.6e")
    (cd / "sparse/0/test.txt").write_text("\n".join(test) + "\n")
    m = (np.abs(lxyz[:, 0] - c[0]) <= a.cell_extent / 2 + 5) & (np.abs(lxyz[:, 1] - c[1]) <= a.cell_extent / 2 + 5)
    arr = np.empty(int(m.sum()), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = lxyz[m].T; arr["nx"] = arr["ny"] = arr["nz"] = 0; arr["red"], arr["green"], arr["blue"] = lrgb[m].T
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(cd / "sparse/0/points3D.ply"))
    print(f"[row-ab] {tag}: {len(ims)} cameras ({len(ims) - len(test)} train + {len(test)} held-out), LiDAR seed {len(arr)} pts, cell centre {np.round(c[:2], 1)}")
build(f"row{a.block}_kf", BL / f"block_{a.block}_ref/transforms.json")
build(f"row{a.block}_full", BL / f"block_{a.block}_full_ref/transforms.json")
