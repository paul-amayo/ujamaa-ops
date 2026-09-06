#!/usr/bin/env python3
"""Top-down sanity check of one ten_rows block: init_lidar.ply vs the block's camera
positions (transforms.json), in the block's own world (camera-world, y DOWN => ground
plane is x-z, height = -y). Prints extents / relative placement; writes a PNG."""
import sys, json
import numpy as np
from pathlib import Path
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BD = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else BD / "init_vs_cameras.png"
tj = json.loads((BD / "transforms.json").read_text())
C = np.array([f["transform_matrix"] for f in tj["frames"]])[:, :3, 3]
ply = PlyData.read(str(BD / tj["ply_file_path"]))
v = ply["vertex"]; P = np.stack([v["x"], v["y"], v["z"]], 1)
h = -P[:, 1]; hc = -C[:, 1]
print(f"[check] {BD.name}: {len(C)} cameras, {len(P):,} init points")
print(f"[check] camera path ptp x/z = {np.ptp(C[:,0]):.1f}/{np.ptp(C[:,2]):.1f} m, cam height (-y) {hc.mean():.2f} m")
print(f"[check] points: x [{P[:,0].min():.1f},{P[:,0].max():.1f}] z [{P[:,2].min():.1f},{P[:,2].max():.1f}] height p2/p50/p98 = {np.percentile(h,2):.2f}/{np.percentile(h,50):.2f}/{np.percentile(h,98):.2f} m (cam = {hc.mean():.2f})")
d = np.linalg.norm(P[:, None, [0, 2]] - C[None, :, [0, 2]], axis=2).min(1) if len(P) < 400_000 else None
if d is not None:
    print(f"[check] point distance to nearest camera (ground plane): median {np.median(d):.2f} m, p90 {np.percentile(d,90):.2f} m, frac within 3 m {np.mean(d<3):.2f}")
fig, ax = plt.subplots(1, 2, figsize=(16, 7), dpi=110)
ax[0].scatter(P[:, 0], P[:, 2], s=0.3, c=h, cmap="viridis", vmin=np.percentile(h, 2), vmax=np.percentile(h, 98))
ax[0].plot(C[:, 0], C[:, 2], "r-", lw=1.2); ax[0].scatter(C[0, 0], C[0, 2], c="lime", s=30, zorder=5); ax[0].set_aspect("equal")
ax[0].set_title(f"{BD.name}: init points (colour = height) + camera path (red)"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("z (m)")
ax[1].hist(h, bins=120, color="#4a8"); ax[1].axvline(hc.mean(), color="r", label=f"camera height {hc.mean():.2f} m"); ax[1].legend()
ax[1].set_title("init point heights (-y)"); ax[1].set_xlabel("height (m)")
fig.tight_layout(); fig.savefig(out); print(f"[check] wrote {out}")
