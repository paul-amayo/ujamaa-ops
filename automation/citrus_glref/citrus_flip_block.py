#!/usr/bin/env python3
"""In-place OpenCV->OpenGL fix for one citrus row-block transforms.json.
build_row_blocks.py wrote OpenCV c2w into the OpenGL contract from 2026-08-13 to 09-06.
Applies T @ diag(1,-1,-1,1), tags pose_convention, keeps a byte backup, idempotent."""
import json, shutil, sys
import numpy as np
from pathlib import Path

bd = Path(sys.argv[1]); p = bd / "transforms.json"
t = json.loads(p.read_text())
if str(t.get("pose_convention", "")).startswith("opengl_c2w"):
    print(f"[flip] {bd.name}: already opengl_c2w — no-op"); sys.exit(0)
T = np.array([f["transform_matrix"] for f in t["frames"]])
tr = T[:, :3, 3]; d = np.diff(tr, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
zdot = float(np.mean(np.sum(T[:-1, :3, 2] * d, 1)))
if zdot < 0.5:
    print(f"[flip] {bd.name}: cam-z·travel {zdot:+.3f} — NOT OpenCV-looking, refusing"); sys.exit(1)
bak = bd / "transforms_cv_pre_glfix.json"
if not bak.exists(): shutil.copy(p, bak)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
for f in t["frames"]:
    f["transform_matrix"] = (np.array(f["transform_matrix"]) @ FLIP).tolist()
t["pose_convention"] = "opengl_c2w"
p.write_text(json.dumps(t, indent=2))
print(f"[flip] {bd.name}: {len(t['frames'])} frames OpenCV->OpenGL (cam-z·travel was {zdot:+.3f}); backup {bak.name}")
