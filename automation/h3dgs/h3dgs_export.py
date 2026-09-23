"""Generic H3DGS export for a UJAMAA survey: all keyframes of a blocks config -> one COLMAP prior model in a
metric, z-up frame + images + interval-10 test split.
  python h3dgs_export.py <survey_root> <proj_dir> [--cfg lio_row100]

Pose convention per block: a `pose_convention` tag starting with `opengl_c2w` is trusted; otherwise it is
MEASURED from motion: for a forward-facing camera the camera's +z column (OpenCV forward / OpenGL backward)
correlates with the velocity, so mean(col2 . v) > 0 means OpenCV-in-OpenGL-contract (the Aug-13..Sep-6 builder
bug) and the matrix is converted with T @ diag(1,-1,-1,1). World up is measured too: the plane normal of the
camera centres, signed by the cameras' up vectors; the world is rotated so that up = +z (chunks cut in x/y).
"""
import argparse, json, os, shutil, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import Camera, Image, write_model, rotmat2qvec

ap = argparse.ArgumentParser()
ap.add_argument("survey_root"); ap.add_argument("proj_dir"); ap.add_argument("--cfg", default="lio_row100")
ap.add_argument("--test_interval", type=int, default=10)
a = ap.parse_args()
SURVEY, PROJ = Path(a.survey_root), Path(a.proj_dir)
BLOCKS = SURVEY / "prod/tassili/blocks_ns" / a.cfg
POSES, IMGS = PROJ / "camera_calibration/poses/sparse/0", PROJ / "camera_calibration/rectified/images"
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

blocks = sorted(x for x in BLOCKS.glob("block_[0-9][0-9][0-9]/transforms.json") if x.parent.name[6:].isdigit())   # canonical blocks only; suffixed dirs are experiments (prod doctrine)
assert blocks, f"no blocks under {BLOCKS}"
frames, intr, conv_counts = [], [], {"tagged_opengl": 0, "measured_opengl": 0, "measured_opencv_flipped": 0}
for tj_path in blocks:
    tj = json.load(open(tj_path)); bid = int(tj_path.parent.name.split("_")[1])
    intr.append([tj[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")])
    for k in ("k1", "k2", "p1", "p2"):
        assert abs(tj.get(k, 0.0)) < 1e-6, f"{tj_path}: distortion {k}={tj.get(k)} — export expects rectified images"
    M = np.array([f["transform_matrix"] for f in tj["frames"]], float)
    tag = str(tj.get("pose_convention", ""))
    if tag.startswith("opengl_c2w"):
        conv_counts["tagged_opengl"] += 1
    else:
        v = np.diff(M[:, :3, 3], axis=0); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        s = float(np.mean(np.sum(M[:-1, :3, 2] * v, axis=1)))
        if s > 0.3:
            M = M @ GL2CV; conv_counts["measured_opencv_flipped"] += 1
        elif s < -0.3:
            conv_counts["measured_opengl"] += 1
        else:
            raise SystemExit(f"{tj_path}: pose convention undecidable from motion (mean col2.v = {s:+.2f})")
    for i, (f, m) in enumerate(zip(tj["frames"], M)):
        p = Path(f["file_path"]); p = p if p.is_absolute() else (tj_path.parent / p)
        frames.append((p.name, p, m, bid, i))
intr = np.array(intr); spread = (intr.max(0) - intr.min(0)) / intr.mean(0)
if spread[:4].max() > 5e-3: print(f"WARNING: per-block intrinsics differ by up to {spread[:4].max()*100:.2f}% — using the median")
fx, fy, cx, cy, w, h = np.median(intr, 0); w, h = int(round(w)), int(round(h))
names = [f[0] for f in frames]; assert len(set(names)) == len(names), "duplicate image names across blocks"
print(f"{len(blocks)} blocks, {len(frames)} keyframes, convention: {conv_counts}, camera fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f} {w}x{h}")

# world up from the camera-centre plane, signed by the cameras' up vectors (OpenGL col 1)
C = np.array([f[2][:3, 3] for f in frames]); U = np.array([f[2][:3, 1] for f in frames])
Cc = C - C.mean(0); evals, evecs = np.linalg.eigh(Cc.T @ Cc)
n = evecs[:, 0]; n = n if U.mean(0) @ n > 0 else -n
x_axis = evecs[:, 2] - (evecs[:, 2] @ n) * n; x_axis /= np.linalg.norm(x_axis); y_axis = np.cross(n, x_axis)
R = np.stack([x_axis, y_axis, n])            # rows: new axes in old coords -> p' = R p
R_W = np.eye(4); R_W[:3, :3] = R
print(f"world up (plane normal, signed by camera up) = {n.round(3)}; mean camera-up . up = {(U @ n).mean():.3f}; det R = {np.linalg.det(R):.3f}")
assert (U @ n).mean() > 0.9 and abs(np.linalg.det(R) - 1) < 1e-6

cams = {1: Camera(id=1, model="PINHOLE", width=w, height=h, params=np.array([fx, fy, cx, cy]))}
images, test_names = {}, []
for k, (name, src, m, bid, i) in enumerate(sorted(frames, key=lambda t: (t[3], t[4]))):
    c2w = R_W @ m @ GL2CV; w2c = np.linalg.inv(c2w)
    images[k + 1] = Image(id=k + 1, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), int))
    if i % a.test_interval == 0: test_names.append(name)
POSES.mkdir(parents=True, exist_ok=True); write_model(cams, images, {}, str(POSES), ".bin")
(POSES / "test.txt").write_text("\n".join(test_names) + "\n")
IMGS.mkdir(parents=True, exist_ok=True); ncopy = 0
for name, src, *_ in frames:
    if not (IMGS / name).exists():
        try: os.link(src, IMGS / name)          # hardlink: same bytes, no extra space (never a symlink)
        except OSError: shutil.copy2(src, IMGS / name)
        ncopy += 1
meta = {"survey_root": str(SURVEY), "blocks_cfg": str(BLOCKS), "n_images": len(images), "n_test": len(test_names), "convention": conv_counts,
        "world_rotation_to_zup": R_W.tolist(), "world_up_in_lio": n.tolist(), "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h},
        "fg_masks": str(SURVEY / "prod/tassili/fg_masks") if (SURVEY / "prod/tassili/fg_masks").exists() else None,
        "pose_convention": "COLMAP w2c (OpenCV) in the z-up frame; c2w_cv = R_W @ c2w_gl @ diag(1,-1,-1,1)"}
json.dump(meta, open(PROJ / "export_meta.json", "w"), indent=2)
print(f"wrote {len(images)} poses, {len(test_names)} test names, copied {ncopy} images -> {PROJ}\nEXPORT DONE")
