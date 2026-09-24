#!/usr/bin/env python
"""Seed an H3DGS chunk with the survey's LiDAR points instead of (or on top of) the sparse COLMAP triangulation.
Reads every canonical block's init_lidar.ply (the per-block fleet's init; LIO world frame), rotates it into the
project's z-up frame (export_meta.world_rotation_to_zup — the global-BA snap keeps that frame to within cm), crops
to the chunk cell plus a margin, subsamples to --max points and writes it as the chunk's sparse/0/points3D.ply
(original kept as points3D_colmap.ply; --keep_colmap merges the two).
  python h3dgs_lidar_init.py <survey_root> <proj_dir> <chunk> [--variant _ref] [--margin 5] [--max 2000000] [--keep_colmap]
"""
import argparse, json, shutil, numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
ap = argparse.ArgumentParser(); ap.add_argument("survey"); ap.add_argument("proj"); ap.add_argument("chunk")
ap.add_argument("--variant", default="", help="block dir suffix to prefer (e.g. _ref)"); ap.add_argument("--cfg", default="lio_row100")
ap.add_argument("--margin", type=float, default=5.0); ap.add_argument("--max", type=int, default=2_000_000); ap.add_argument("--keep_colmap", action="store_true")
a = ap.parse_args(); S, P = Path(a.survey), Path(a.proj); BL = S / "prod/tassili/blocks_ns" / a.cfg
R_W = np.array(json.load(open(P / "export_meta.json"))["world_rotation_to_zup"])[:3, :3]   # stored as a 4x4
cd = P / "camera_calibration/chunks" / a.chunk; c = np.loadtxt(cd / "center.txt"); e = np.loadtxt(cd / "extent.txt")
lo, hi = c[:2] - e[:2] / 2 - a.margin, c[:2] + e[:2] / 2 + a.margin
blocks = sorted(x for x in BL.glob("block_[0-9][0-9][0-9]") if x.name[6:].isdigit())
pts, cols, used = [], [], 0
for b in blocks:
    src = b.parent / (b.name + a.variant) if a.variant and (b.parent / (b.name + a.variant) / "init_lidar.ply").exists() else b
    f = src / "init_lidar.ply"
    if not f.exists(): continue
    v = PlyData.read(str(f))["vertex"]; xyz = (R_W @ np.stack([v["x"], v["y"], v["z"]], 1).T).T
    m = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
    if not m.any(): continue
    pts.append(xyz[m]); used += 1
    names = v.data.dtype.names
    cols.append(np.stack([v[k] for k in ("red", "green", "blue")], 1)[m] if "red" in names else np.full((int(m.sum()), 3), 128, np.uint8))
xyz = np.concatenate(pts); rgb = np.concatenate(cols).astype(np.uint8)
print(f"[lidar-init] chunk {a.chunk}: {len(xyz)} LiDAR points inside the cell+{a.margin:g} m from {used}/{len(blocks)} blocks (variant {a.variant!r})")
if len(xyz) > a.max:
    sel = np.random.default_rng(0).choice(len(xyz), a.max, replace=False); xyz, rgb = xyz[sel], rgb[sel]; print(f"[lidar-init] subsampled to {a.max}")
ply = cd / "sparse/0/points3D.ply"; bak = cd / "sparse/0/points3D_colmap.ply"
if not bak.exists():
    if ply.exists(): shutil.copy(ply, bak)
    else:   # H3DGS only writes points3D.ply from points3D.bin at first training; convert the COLMAP seed here so the backup exists
        import sys; sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_points3D_binary
        pc = read_points3D_binary(str(cd / "sparse/0/points3D.bin")); cx = np.array([q.xyz for q in pc.values()]); cc = np.array([q.rgb for q in pc.values()]).astype(np.uint8)
        arr0 = np.empty(len(cx), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr0["x"], arr0["y"], arr0["z"] = cx.T; arr0["nx"] = arr0["ny"] = arr0["nz"] = 0; arr0["red"], arr0["green"], arr0["blue"] = cc.T
        PlyData([PlyElement.describe(arr0, "vertex")]).write(str(bak))
if a.keep_colmap:
    v = PlyData.read(str(bak))["vertex"]; cx = np.stack([v["x"], v["y"], v["z"]], 1); cc = np.stack([v[k] for k in ("red", "green", "blue")], 1).astype(np.uint8)
    xyz, rgb = np.concatenate([xyz, cx]), np.concatenate([rgb, cc]); print(f"[lidar-init] + {len(cx)} COLMAP points")
n = np.zeros_like(xyz)
arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
arr["x"], arr["y"], arr["z"] = xyz.T; arr["nx"], arr["ny"], arr["nz"] = n.T; arr["red"], arr["green"], arr["blue"] = rgb.T
PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply)); print(f"[lidar-init] wrote {ply} ({len(arr)} points); COLMAP copy at {bak.name}")
