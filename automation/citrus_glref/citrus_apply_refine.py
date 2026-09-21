#!/usr/bin/env python3
"""Sim(3)-align the block's GLOMAP poses (colmap_glref/transforms.json, GL) into the block's
LIO world and write them into the block's transforms.json IN PLACE (odometry backup kept).
Guards: coverage >=90%, alignment p50 <=0.5 m, else refuses and leaves odometry poses."""
import json, shutil, sys
import numpy as np
from pathlib import Path

bd = Path(sys.argv[1]); W = bd / "colmap_glref"
t = json.loads((bd / "transforms.json").read_text())
assert t.get("pose_convention", "").startswith("opengl"), "flip to opengl_c2w first"
if "COLMAP" in t.get("pose_convention", ""):
    print(f"[apply] {bd.name}: already refined — no-op"); sys.exit(0)
cm = {Path(f["file_path"]).name: np.array(f["transform_matrix"])
      for f in json.loads((W / "transforms.json").read_text())["frames"]}
ours = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in t["frames"]}
names = [n for n in ours if n in cm]
cov = len(names) / len(ours)
P = np.array([cm[n][:3, 3] for n in names]); Q = np.array([ours[n][:3, 3] for n in names])
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq
U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = (U @ D @ Vt).T; s = (S * np.diag(D)).sum() / (X ** 2).sum(); tt = mq - s * R @ mp
res = np.linalg.norm(Q - (s * (R @ P.T).T + tt), axis=1); p50 = float(np.percentile(res, 50))
if cov < 0.9 or p50 > 0.5:
    print(f"[apply] {bd.name}: REFUSED — coverage {cov:.0%}, p50 {p50:.3f} m (need >=90% / <=0.5 m); odometry poses kept")
    sys.exit(1)
bak = bd / "transforms_odo_glfix.json"
if not bak.exists(): shutil.copy(bd / "transforms.json", bak)
frames = []
for f in t["frames"]:
    n = Path(f["file_path"]).name
    if n not in cm: continue          # unregistered frame: drop rather than mix pose sources
    C = cm[n]; Tn = np.eye(4); Tn[:3, :3] = R @ C[:3, :3]; Tn[:3, 3] = s * R @ C[:3, 3] + tt
    frames.append(dict(f, transform_matrix=Tn.tolist()))
t["frames"] = frames
t["pose_convention"] = "opengl_c2w (COLMAP/GLOMAP Sim3-aligned into LIO world)"
(bd / "transforms.json").write_text(json.dumps(t, indent=2))
print(f"[apply] {bd.name}: {len(frames)}/{len(ours)} frames refined; Sim(3) scale {s:.3f}, p50 {p50:.3f} m p90 {np.percentile(res,90):.3f} m; odometry backup {bak.name}")
