"""Refined-pose block: COLMAP poses (colmap_<blk>/transforms.json, GL) Sim(3)-aligned into the
block's current (roll-corrected) world using camera positions -> block_<NNN>_ref/transforms.json.
Keeps our metric scale/world (LiDAR init stays valid), replaces per-frame poses with COLMAP's."""
import sys, json, shutil, numpy as np
from pathlib import Path
blk = sys.argv[1]; T = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili"); B = T/"blocks_ns/lio_row100"
W = T/("colmap_b013" if blk == "block_013" else f"colmap_{blk}"); FLIP = np.diag([1., -1., -1., 1.])
cm = json.loads((W/"transforms.json").read_text()); ours = json.loads((B/blk/"transforms.json").read_text())
cmap = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in cm["frames"]}            # GL c2w, COLMAP world
omap = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in ours["frames"]}          # GL c2w, our world
names = [Path(f["file_path"]).name for f in ours["frames"] if Path(f["file_path"]).name in cmap]
P = np.array([cmap[n][:3, 3] for n in names]); Q = np.array([omap[n][:3, 3] for n in names])
mp, mq = P.mean(0), Q.mean(0); X, Y = P - mp, Q - mq; U, S, Vt = np.linalg.svd(X.T @ Y); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rr = (U @ D @ Vt).T; s = (S * np.diag(D)).sum() / (X ** 2).sum(); t = mq - s * Rr @ mp
res = np.linalg.norm(Q - (s * (Rr @ P.T).T + t), axis=1)
out = {k: v for k, v in ours.items() if k != "frames"}; out["pose_convention"] = "opengl_c2w (COLMAP poses Sim3-aligned into the LIO world)"; frames = []
for f in ours["frames"]:
    n = Path(f["file_path"]).name
    if n not in cmap: continue
    C = cmap[n]; Tn = np.eye(4); Tn[:3, :3] = Rr @ C[:3, :3]; Tn[:3, 3] = s * Rr @ C[:3, 3] + t
    frames.append(dict(f, transform_matrix=Tn.tolist()))
out["frames"] = frames
for k in ("train_filenames", "val_filenames", "test_filenames"):
    if k in ours: out[k] = [x for x in ours[k] if Path(x).name in cmap]
bd = B/f"{blk}_ref"; bd.mkdir(exist_ok=True); (bd/"transforms.json").write_text(json.dumps(out, indent=2))
if (B/blk/"init_lidar.ply").exists(): shutil.copy(B/blk/"init_lidar.ply", bd/"init_lidar.ply")
print(f"[ref] {blk}: {len(frames)}/{len(ours['frames'])} frames refined; Sim(3) scale {s:.3f}, position p50 {np.percentile(res,50):.3f} m p90 {np.percentile(res,90):.3f} m; init = block's LiDAR init")
