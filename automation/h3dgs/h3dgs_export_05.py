"""Export 05_13D_Jackal (prod lio_row100 blocks, OpenGL c2w in the LIO world) as a Hierarchical-3DGS
project: a COLMAP prior model with ALL 3386 keyframes in one metric, z-up frame, the keyframe images,
and the held-out list (test.txt = every 10th keyframe per block, nerfstudio `interval 10` semantics).

Frames: LIO world is y-DOWN (measured mean camera-up = (-0.02,-0.99,0.02)); H3DGS chunks cut in x/y and
treat z as up, so the world is rotated by  x'=x, y'=z, z'=-y  (right-handed, metric kept).
Poses stay in OpenCV w2c for COLMAP (OpenGL c2w -> post-multiply diag(1,-1,-1,1) -> invert).
"""
import json, sys, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import Camera, Image, write_model, rotmat2qvec

SURVEY = Path("/home/paperspace/data/citrus_all/05_13D_Jackal")
BLOCKS = SURVEY / "prod/tassili/blocks_ns/lio_row100"
PROJ = SURVEY / "experimental/h3dgs"
POSES = PROJ / "camera_calibration/poses/sparse/0"
IMGS = PROJ / "camera_calibration/rectified/images"
R_W = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # LIO (y-down) -> z-up
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

blocks = sorted(BLOCKS.glob("block_*/transforms.json"))
intr = []
frames = []          # (name, src_path, c2w_gl, block_id, idx_in_block)
for tj_path in blocks:
    tj = json.load(open(tj_path))
    bid = int(tj_path.parent.name.split("_")[1])
    intr.append([tj[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")])
    assert tj.get("pose_convention", "").startswith("opengl_c2w"), (tj_path, tj.get("pose_convention"))
    for i, f in enumerate(tj["frames"]):
        frames.append((Path(f["file_path"]).name, Path(f["file_path"]), np.array(f["transform_matrix"], float), bid, i))
intr = np.array(intr)
spread = (intr.max(0) - intr.min(0)) / intr.mean(0)
print(f"blocks {len(blocks)} frames {len(frames)} unique {len(set(f[0] for f in frames))}")
print("intrinsics mean", intr.mean(0).round(3), "relative spread", spread.round(5))
assert spread[:4].max() < 5e-3, "per-block intrinsics differ by >0.5% — export needs per-block cameras"
fx, fy, cx, cy, w, h = intr.mean(0); w, h = int(round(w)), int(round(h))

cams = {1: Camera(id=1, model="PINHOLE", width=w, height=h, params=np.array([fx, fy, cx, cy]))}
images = {}
ups = []
test_names = []
for k, (name, src, c2w_gl, bid, i) in enumerate(sorted(frames, key=lambda t: (t[3], t[4]))):
    c2w = R_W @ c2w_gl @ GL2CV               # z-up world, OpenCV camera
    ups.append((R_W @ c2w_gl)[:3, 1])        # OpenGL up column after the world rotation
    w2c = np.linalg.inv(c2w)
    images[k + 1] = Image(id=k + 1, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1,
                          name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), int))
    if i % 10 == 0:
        test_names.append(name)
ups = np.array(ups)
print("mean camera up after rotation (want +z):", ups.mean(0).round(3))
assert ups.mean(0)[2] > 0.9

POSES.mkdir(parents=True, exist_ok=True)
write_model(cams, images, {}, str(POSES), ".bin")
(POSES / "test.txt").write_text("\n".join(test_names) + "\n")
json.dump({"world_rotation_lio_to_h3dgs": R_W.tolist(), "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h},
           "n_images": len(images), "n_test": len(test_names), "source_blocks_cfg": str(BLOCKS),
           "pose_convention": "COLMAP w2c (OpenCV) in the z-up frame; c2w_cv = R_W @ c2w_gl @ diag(1,-1,-1,1)"},
          open(PROJ / "export_meta.json", "w"), indent=2)
print(f"wrote {len(images)} images, {len(test_names)} test names -> {POSES}")

IMGS.mkdir(parents=True, exist_ok=True)
n = 0
for name, src, *_ in frames:
    dst = IMGS / name
    if not dst.exists():
        shutil.copy2(src, dst); n += 1
print(f"copied {n} images -> {IMGS} (total {len(list(IMGS.glob('*.png')))})")
print("EXPORT DONE")
