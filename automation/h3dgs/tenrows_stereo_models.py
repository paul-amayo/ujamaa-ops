"""COLMAP model helpers for the ten_rows stereo experiment (2026-09-25): add the ZED right-camera keyframes to an H3DGS
chunk model with poses derived from the left ones.
  left <db.db> <chunk sparse/0> <out model dir>
      the chunk's LEFT poses re-keyed to the database's image ids (camera 1 = the database camera), no points:
      the input model for point_triangulator / mapper --fix_existing_frames.
  rig  <registered model dir> <chunk sparse/0> <out dir>
      per-frame left->right transform from the registered right images (right camera centre in the left camera frame,
      rotation angle), the median rig transform, and <out>/sparse/0 = chunk model + right images at
      w2c_R = T_rig @ w2c_L (no 2D points), <out>/sparse/0/test.txt = chunk test names + their right pairs.
Run with the h3dgs env python (read_write_model from the H3DGS preprocess dir)."""
import json, os, re, shutil, sqlite3, sys
import numpy as np
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_model, write_model, Camera, Image, qvec2rotmat, rotmat2qvec

def w2c_of(im):
    T = np.eye(4); T[:3, :3] = qvec2rotmat(im.qvec); T[:3, 3] = im.tvec; return T

def image_at(id_, name, T, cam_id):
    return Image(id=id_, qvec=rotmat2qvec(T[:3, :3]), tvec=T[:3, 3], camera_id=cam_id, name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64))

mode = sys.argv[1]
if mode == "left":
    db, chunk, out = sys.argv[2:5]; os.makedirs(out, exist_ok=True)
    con = sqlite3.connect(db); ids = {n: i for i, n in con.execute("SELECT image_id, name FROM images")}
    cam_rows = list(con.execute("SELECT camera_id, model, width, height, params FROM cameras")); con.close()
    cams, ims, _ = read_model(chunk, ".bin")
    cid, model, w, h, params = cam_rows[0]; params = np.frombuffer(params, np.float64)
    CAM = {"0": "SIMPLE_PINHOLE", "1": "PINHOLE"}.get(str(model), "PINHOLE")
    ccam = list(cams.values())[0]
    assert np.allclose(params, ccam.params, atol=1e-3), f"database camera {params} != chunk camera {ccam.params}"
    cams_out = {cid: Camera(id=cid, model=CAM, width=w, height=h, params=params)}
    ims_out = {}
    for im in ims.values():
        if im.name in ids: ims_out[ids[im.name]] = image_at(ids[im.name], im.name, w2c_of(im), cid)
    write_model(cams_out, ims_out, {}, out, ".bin")
    print(f"[stereo-models] left model: {len(ims_out)}/{len(ims)} chunk images matched to database ids ({len(ids)} images in the database, {len(ids) - len(ims_out)} to register)", flush=True)
elif mode == "rig":
    reg, chunk, out = sys.argv[2:5]
    if not os.path.exists(os.path.join(reg, "images.bin")) and os.path.exists(os.path.join(reg, "0", "images.bin")): reg = os.path.join(reg, "0")
    cams, rims, rpts = read_model(reg, ".bin"); ccams, cims, cpts = read_model(chunk, ".bin")
    left = {im.name: im for im in cims.values()}; rows = []
    for im in rims.values():
        m = re.match(r"(kf_\d+)_R\.png$", im.name)
        if not m or m.group(1) + ".png" not in left: continue
        L = left[m.group(1) + ".png"]; TL, TR = w2c_of(L), w2c_of(im)
        T = TR @ np.linalg.inv(TL)                      # left-camera frame -> right-camera frame
        c_r_in_l = -T[:3, :3].T @ T[:3, 3]              # right centre expressed in the left camera frame
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        rows.append((im.name, c_r_in_l, ang, T, len(im.point3D_ids[im.point3D_ids >= 0])))
    if not rows: print("[stereo-models] no right images registered"); sys.exit(1)
    C = np.array([r[1] for r in rows]); A = np.array([r[2] for r in rows]); npts = np.array([r[4] for r in rows])
    med = np.median(C, 0); spread = np.percentile(np.linalg.norm(C - med, axis=1), [50, 90, 100])
    print(f"[stereo-models] {len(rows)} right images registered (of {sum(1 for n in left if True)} left frames); right centre in the left camera frame: median {np.round(med, 4).tolist()} units, "
          f"p10 {np.round(np.percentile(C, 10, 0), 4).tolist()} p90 {np.round(np.percentile(C, 90, 0), 4).tolist()}; |offset from median| p50 {spread[0]:.4f} p90 {spread[1]:.4f} max {spread[2]:.4f}; "
          f"rotation angle median {np.median(A):.3f} deg p90 {np.percentile(A, 90):.3f}; 3D points per right image median {int(np.median(npts))}", flush=True)
    # rig transform: median translation, rotation = identity if the registered rotations are within 0.2 deg (rectified pair), else the chordal-mean rotation
    if np.median(A) < 0.2: R = np.eye(3); rot_note = "identity (rectified pair)"
    else:
        M = np.mean([r[3][:3, :3] for r in rows], 0); U, _, Vt = np.linalg.svd(M); R = U @ Vt
        if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
        rot_note = f"chordal mean ({np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.3f} deg)"
    T_rig = np.eye(4); T_rig[:3, :3] = R; T_rig[:3, 3] = -R @ med
    print(f"[stereo-models] rig transform: translation {np.round(med, 4).tolist()} (|b| {np.linalg.norm(med):.4f} units), rotation {rot_note}", flush=True)
    sp = os.path.join(out, "sparse", "0"); os.makedirs(sp, exist_ok=True)
    ims_out = dict(cims); nid = max(cims) + 1; cid = list(ccams)[0]
    for name, L in sorted(left.items()):
        ims_out[nid] = image_at(nid, name[:-4] + "_R.png", T_rig @ w2c_of(L), cid); nid += 1
    write_model(ccams, ims_out, cpts, sp, ".bin")
    test = [l.strip() for l in open(os.path.join(chunk, "test.txt")) if l.strip()]
    with open(os.path.join(sp, "test.txt"), "w") as f: f.write("\n".join(test + [t[:-4] + "_R.png" for t in test]) + "\n")
    for extra in ("depth_params.json", "points3D_colmap.ply", "points3D.ply"):
        if os.path.exists(os.path.join(chunk, extra)): shutil.copy2(os.path.join(chunk, extra), sp)
    json.dump({"T_rig_left_to_right": T_rig.tolist(), "baseline_units": float(np.linalg.norm(med)), "n_registered": len(rows), "rotation": rot_note}, open(os.path.join(out, "rig.json"), "w"), indent=1)
    print(f"[stereo-models] wrote {sp}: {len(cims)} left + {len(ims_out) - len(cims)} right images, test.txt {len(test)} + {len(test)} names", flush=True)
