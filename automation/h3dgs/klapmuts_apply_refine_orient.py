#!/usr/bin/env python3
"""Sim(3)-align a block's GLOMAP poses (<work>/transforms.json, OpenGL c2w) onto the block's odometry transforms
(<base>, OpenGL c2w) — ORIENTATION-AWARE variant of klapmuts_apply_refine_named.py: the rotation is the chordal mean of
the per-frame relative rotations (odometry orientation x SfM orientation^-1), which pins the roll about a straight
lane that camera centres alone leave to noise (block_003: 8 deg); scale and translation come from the centres. Same
guards (coverage >= 90 %, alignment p50 <= 0.5 m); the block's transforms.json is never touched.
  python klapmuts_apply_refine_orient.py <block_dir> <work_dir> <base transforms name> <out transforms name>"""
import json, sys
import numpy as np
from pathlib import Path
bd, W, base, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
t = json.loads((bd / base).read_text()); assert t.get("pose_convention", "").startswith("opengl"), f"{base}: not opengl_c2w"
cm = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in json.loads((W / "transforms.json").read_text())["frames"]}
ours = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in t["frames"]}
names = [n for n in ours if n in cm]; cov = len(names) / len(ours)
P = np.array([cm[n][:3, 3] for n in names]); Q = np.array([ours[n][:3, 3] for n in names])
# 1. centre-based Umeyama (positions optimal, pitch/yaw well determined; the roll about a straight lane is not)
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq
U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rc = (U @ D @ Vt).T; s = (S * np.diag(D)).sum() / (X ** 2).sum()
# 2. roll about the lane axis from the camera orientations: chordal mean of (odometry orientation x centre-aligned SfM
#    orientation^-1), projected on the lane axis; pitch/yaw stay centre-based (the orientation targets carry the rig's
#    laser->camera extrinsic error, which would otherwise displace the block ends and open the seams)
axis = Rc @ np.linalg.svd(X, full_matrices=False)[2][0]; axis /= np.linalg.norm(axis)
E = np.mean([ours[n][:3, :3] @ (Rc @ cm[n][:3, :3]).T for n in names], 0); Ue, _, Vte = np.linalg.svd(E); Re = Ue @ Vte
if np.linalg.det(Re) < 0: Ue[:, -1] *= -1; Re = Ue @ Vte
rv = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]); ang = np.arctan2(np.linalg.norm(rv) / 2, (np.trace(Re) - 1) / 2)
rv = rv / (np.linalg.norm(rv) + 1e-12) * ang; roll = float(rv @ axis)
K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]); Rroll = np.eye(3) + np.sin(roll) * K + (1 - np.cos(roll)) * K @ K
R = Rroll @ Rc; tt = mq - s * R @ mp
res = np.linalg.norm(Q - (s * (R @ P.T).T + tt), axis=1); p50 = float(np.percentile(res, 50))
dev = np.degrees([np.arccos(np.clip((np.trace(R.T @ ours[n][:3, :3] @ cm[n][:3, :3].T) - 1) / 2, -1, 1)) for n in names])
if cov < 0.9 or p50 > 0.5:
    print(f"[apply-orient] {bd.name}: REFUSED — coverage {cov:.0%}, p50 {p50:.3f} m (need >=90% / <=0.5 m); {out} not written"); sys.exit(1)
frames = []
for f in t["frames"]:
    n = Path(f["file_path"]).name
    if n not in cm: continue
    C = cm[n]; Tn = np.eye(4); Tn[:3, :3] = R @ C[:3, :3]; Tn[:3, 3] = s * R @ C[:3, 3] + tt; frames.append(dict(f, transform_matrix=Tn.tolist()))
t["frames"] = frames; t["pose_convention"] = "opengl_c2w (COLMAP/GLOMAP fixed-camera, Sim3-aligned onto the odometry: rotation from the camera orientations, scale/translation from the centres)"
(bd / out).write_text(json.dumps(t, indent=2))
print(f"[apply-orient] {bd.name}: {len(frames)}/{len(ours)} frames; scale {s:.3f}, centre residual p50 {p50:.3f} p90 {np.percentile(res, 90):.3f} m; orientation residual median {np.median(dev):.2f} p90 {np.percentile(dev, 90):.2f} deg -> {out}")
