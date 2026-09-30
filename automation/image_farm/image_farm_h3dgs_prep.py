"""Single-chunk H3DGS project for one image_farm phone segment (2026-09-30, Paul: "hd3gs is demo standard" — the
phone surveys need H3DGS to be shown). Mirrors automation/tenrows/tenrows_lane_prep.py `chunk` mode, with the
segment's own SfM as the pose source instead of LiDAR odometry:

  <segment>/<proj>/camera_calibration/chunks/lane/sparse/0   COLMAP bin model, one PINHOLE camera, poses from the
                                                             segment's transforms.json (OpenGL c2w -> COLMAP w2c);
                                                             points3D.ply = the segment's COLMAP-sparse init.ply
  <segment>/<proj>/camera_calibration/rectified/images       frames UNDISTORTED with the segment's OPENCV k1 k2 p1 p2
                                                             (same K — H3DGS renders pinhole)
  center/extent = camera bbox + 6 m each side; test.txt = every 10th frame by frame index; aligned/sparse/0 +
  export_meta.json so h3dgs_eval_chunk.py runs.

Why not the segment's sparse/0: it is written by COLMAP 3.14-dev (frames.bin / rigs.bin), and the pose source
of record for these segments is transforms.json (image_farm_prep.py -> colmap_to_nerfstudio.py).
  usage: image_farm_h3dgs_prep.py <segment dir> [--proj h3dgs] [--margin 6]"""
import argparse, json, re, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import write_model, Camera, Image, rotmat2qvec
ap = argparse.ArgumentParser(); ap.add_argument("segment"); ap.add_argument("--proj", default="h3dgs"); ap.add_argument("--margin", type=float, default=6.0)
a = ap.parse_args(); S = Path(a.segment).resolve(); P = S / a.proj; CC = P / "camera_calibration"; CH = CC / "chunks/lane"; SP = CH / "sparse/0"
for d in (SP, CC / "rectified/images", CC / "aligned/sparse/0", P / "output/trained_chunks"): d.mkdir(parents=True, exist_ok=True)
J = json.load(open(S / "transforms.json")); GL = np.diag([1.0, -1.0, -1.0, 1.0])
W, H = int(J["w"]), int(J["h"]); K = np.array([[J["fl_x"], 0, J["cx"]], [0, J["fl_y"], J["cy"]], [0, 0, 1]], np.float64)
dist = np.array([J.get("k1", 0), J.get("k2", 0), J.get("p1", 0), J.get("p2", 0)], np.float64)
idx = lambda f: int(re.findall(r"\d+", Path(f["file_path"]).name)[-1])
frames = sorted(J["frames"], key=idx); names = [Path(f["file_path"]).name for f in frames]; test = names[::10]
cams = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))}; ims = {}; C = []
for i, f in enumerate(frames, 1):
    name = Path(f["file_path"]).name; c2w = np.asarray(f["transform_matrix"], np.float64)
    if c2w.shape[0] == 3: c2w = np.vstack([c2w, [0, 0, 0, 1]])
    c2w = c2w @ GL; w2c = np.linalg.inv(c2w); C.append(c2w[:3, 3])
    ims[i] = Image(id=i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64))
    dst = CC / "rectified/images" / name
    if not dst.exists():
        im = cv2.imread(str(S / f["file_path"]), cv2.IMREAD_UNCHANGED)
        assert im is not None and im.shape[:2] == (H, W), (name, None if im is None else im.shape)
        cv2.imwrite(str(dst), cv2.undistort(im, K, dist, None, K))
write_model(cams, ims, {}, str(SP), ".bin"); C = np.array(C); ctr = (C.max(0) + C.min(0)) / 2; ext = (C.max(0) - C.min(0)) + 2 * a.margin
np.savetxt(CH / "center.txt", ctr); np.savetxt(CH / "extent.txt", ext)
(SP / "test.txt").write_text("\n".join(test) + "\n"); (CC / "aligned/sparse/0/test.txt").write_text("\n".join(test) + "\n"); write_model(cams, ims, {}, str(CC / "aligned/sparse/0"), ".bin")
ply = S / J.get("ply_file_path", "init.ply")
if ply.exists(): shutil.copy2(ply, SP / "points3D.ply")
json.dump({"survey_root": str(S), "n_images": len(ims), "n_test": len(test), "n_train": len(ims) - len(test), "camera": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "w": W, "h": H},
           "undistorted_from": {"model": "OPENCV", "k1_k2_p1_p2": dist.tolist()}, "sky_masks": None, "fg_masks": None, "world_rotation_to_zup": np.eye(4).tolist(),
           "pose_convention": "COLMAP w2c (OpenCV) in the segment's SfM world (transforms.json)"}, open(P / "export_meta.json", "w"), indent=1)
print(f"[if-h3dgs-prep] {S.name}/{a.proj}: {len(ims)} images ({len(ims) - len(test)} training, {len(test)} held-out), {W}x{H} undistorted, "
      f"cell centre {ctr.round(2).tolist()} extent {ext.round(1).tolist()}, init {'yes' if ply.exists() else 'NO'}", flush=True)
