"""Chunk-level dataset for an H3DGS containment side-car (Paul, 2026-09-27: "why blocks and not chunks?"): a nerfstudio
block-style dir for one H3DGS chunk — transforms.json from the chunk's own (chunk-BA) camera poses converted back to the
survey's LIO world in the OpenGL c2w contract, with the chunk's single PINHOLE camera, and supervision/trees_only = the
union of the prod blocks' compiled id maps for those keyframes (hardlinks; a keyframe painted in two blocks keeps the
first). Keyframe images stay where the blocks point (prod/scratch_sam3).
  python (h3dgs env) sidecar_chunk_dataset.py <survey id> <chunk> <out dir> [--cfg lio_row100]"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser(); ap.add_argument("survey"); ap.add_argument("chunk"); ap.add_argument("out"); ap.add_argument("--cfg", default="lio_row100"); a = ap.parse_args()
S = Path("/home/paperspace/data/citrus_all") / a.survey; P = S / "experimental/h3dgs"; C = P / "camera_calibration/chunks" / a.chunk; O = Path(a.out); (O / "supervision/trees_only").mkdir(parents=True, exist_ok=True)
meta = json.load(open(P / "export_meta.json")); R_W = np.asarray(meta.get("world_rotation_to_zup") or meta["world_rotation_lio_to_h3dgs"], np.float64); GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
cam = read_cameras_binary(str(C / "sparse/0/cameras.bin")); cam = cam[sorted(cam)[0]]; fx, fy, cx, cy = [float(v) for v in cam.params[:4]]
# where the block keyframes live (any block's transforms.json gives the image root) and the supervision maps per keyframe
root = None; sup = {}
for tj in sorted(glob.glob(str(S / "prod/tassili/blocks_ns" / a.cfg / "block_[0-9][0-9][0-9]" / "transforms.json"))):
    bd = Path(tj).parent; t = json.load(open(tj))
    if root is None and t["frames"]: root = Path(t["frames"][0]["file_path"]).parent
    for f in sorted(glob.glob(str(bd / "supervision/trees_only/kf_*.png"))): sup.setdefault(os.path.basename(f), f)
ims = read_images_binary(str(C / "sparse/0/images.bin")); frames = []; n_sup = 0
for im in sorted(ims.values(), key=lambda im: im.name):
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w_h = np.linalg.inv(w2c)
    c2w_lio = np.eye(4); c2w_lio[:3, :3] = R_W[:3, :3].T @ c2w_h[:3, :3]; c2w_lio[:3, 3] = R_W[:3, :3].T @ c2w_h[:3, 3]
    frames.append({"file_path": str(root / im.name), "transform_matrix": (c2w_lio @ GL2CV).tolist()})
    if im.name in sup:
        dst = O / "supervision/trees_only" / im.name
        if not dst.exists(): os.link(sup[im.name], dst)
        n_sup += 1
J = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": int(cam.width), "h": int(cam.height), "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV",
     "pose_convention": f"opengl_c2w (H3DGS chunk {a.chunk} chunk-BA poses mapped back into the LIO world)", "frames": frames}
(O / "transforms.json").write_text(json.dumps(J, indent=1))
print(f"[chunk-dataset] {a.chunk}: {len(frames)} cameras ({n_sup} with supervision maps), camera {fx:.1f}/{fy:.1f} @ {cx:.1f},{cy:.1f} {cam.width}x{cam.height} -> {O}", flush=True)
