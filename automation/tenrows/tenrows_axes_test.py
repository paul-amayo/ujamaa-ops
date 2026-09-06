"""Which camera-axis convention do the block poses follow? Project init_lidar.ply into a
training keyframe under (a) OpenCV c2w (x right, y down, z forward) and (b) OpenGL c2w
(x right, y up, z back) and overlay on the image. The matching overlay = the data's convention."""
import json, sys, numpy as np
from pathlib import Path
from plyfile import PlyData
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BD = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100/block_013")
tj = json.loads((BD/"transforms.json").read_text()); fx, fy, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
fr = tj["frames"][len(tj["frames"])//2]; T = np.array(fr["transform_matrix"]); img = np.asarray(Image.open(fr["file_path"]).convert("RGB"))
v = PlyData.read(str(BD/"init_lidar.ply"))["vertex"]; P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
def proj(c2w_cv):
    pc = (np.linalg.inv(c2w_cv) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; z = pc[:, 2]; ok = z > 0.3
    u = fx*pc[ok,0]/z[ok] + cx; vv = fy*pc[ok,1]/z[ok] + cy; inside = (u>=0)&(u<1280)&(vv>=0)&(vv<720)
    return u[inside], vv[inside], z[ok][inside], inside.sum(), ok.sum()
FLIP = np.diag([1, -1, -1, 1])
cases = [("as OpenCV c2w (x right, y down, z fwd)", T), ("as OpenGL c2w (x right, y up, z back) -> cv", T @ FLIP)]
fig, ax = plt.subplots(1, 2, figsize=(18, 5.6), dpi=110)
for a, (name, C) in zip(ax, cases):
    u, vv, z, nin, nfront = proj(C); a.imshow(img); a.scatter(u, vv, s=0.6, c=np.clip(z, 0, 15), cmap="turbo", alpha=0.6)
    a.set_title(f"{name}: {nfront:,} in front, {nin:,} in image", fontsize=10); a.set_axis_off()
    print(f"[axes] {name}: points in front of camera {nfront:,}, inside image {nin:,}, median depth {np.median(z) if len(z) else float('nan'):.2f} m")
fig.suptitle(f"block_013 {Path(fr['file_path']).name}: LiDAR init projected with the block's transform_matrix under two axis conventions"); fig.tight_layout()
out = "/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/block_013_axes_test.png"; fig.savefig(out); print("[axes] wrote", out)
