#!/usr/bin/env python3
"""Sim(3)-align a block's GLOMAP poses (<work>/transforms.json, OpenGL c2w) into the block's LIO world, using the
ODOMETRY transforms as the alignment target, and write the result to a NAMED file — the block's transforms.json is
never touched. Guards as citrus_apply_refine.py (coverage >= 90 %, alignment p50 <= 0.5 m).
  python klapmuts_apply_refine_named.py <block_dir> <work_dir> <base transforms name> <out transforms name>"""
import json, sys
import numpy as np
from pathlib import Path
bd, W, base, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
t = json.loads((bd / base).read_text())
assert t.get("pose_convention", "").startswith("opengl"), f"{base}: not opengl_c2w"
cm = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in json.loads((W / "transforms.json").read_text())["frames"]}
ours = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in t["frames"]}
names = [n for n in ours if n in cm]; cov = len(names) / len(ours)
P = np.array([cm[n][:3, 3] for n in names]); Q = np.array([ours[n][:3, 3] for n in names])
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq
U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = (U @ D @ Vt).T; s = (S * np.diag(D)).sum() / (X ** 2).sum(); tt = mq - s * R @ mp
res = np.linalg.norm(Q - (s * (R @ P.T).T + tt), axis=1); p50 = float(np.percentile(res, 50))
if cov < 0.9 or p50 > 0.5:
    print(f"[apply] {bd.name}: REFUSED — coverage {cov:.0%}, p50 {p50:.3f} m (need >=90% / <=0.5 m); {out} not written"); sys.exit(1)
frames = []
for f in t["frames"]:
    n = Path(f["file_path"]).name
    if n not in cm: continue
    C = cm[n]; Tn = np.eye(4); Tn[:3, :3] = R @ C[:3, :3]; Tn[:3, 3] = s * R @ C[:3, 3] + tt
    frames.append(dict(f, transform_matrix=Tn.tolist()))
t["frames"] = frames; t["pose_convention"] = "opengl_c2w (COLMAP/GLOMAP fixed-camera, Sim3-aligned into LIO world)"
(bd / out).write_text(json.dumps(t, indent=2))
print(f"[apply] {bd.name}: {len(frames)}/{len(ours)} frames refined; Sim(3) scale {s:.3f}, p50 {p50:.3f} m p90 {np.percentile(res,90):.3f} m -> {out}")
