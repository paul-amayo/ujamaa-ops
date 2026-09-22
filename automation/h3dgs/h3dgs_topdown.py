"""Top-down of the H3DGS chunk grid over the 05_13D survey: camera trajectory coloured by the OLD lio_row100
block id (with the block hand-offs marked), the six 30 m chunk cells, and per-chunk training-camera sets
(inside the cell = filled, pulled in by visibility = hollow)."""
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_images_binary, qvec2rotmat
CC = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/camera_calibration")
BL = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
ims = read_images_binary(str(CC / "aligned/sparse/0/images.bin"))
centre = {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()}
block_of, order = {}, []
for tj in sorted(BL.glob("block_*/transforms.json")):
    b = int(tj.parent.name.split("_")[1])
    for f in json.load(open(tj))["frames"]:
        n = Path(f["file_path"]).name; block_of[n] = b; order.append(n)
P = np.array([centre[n] for n in order]); B = np.array([block_of[n] for n in order])
chunks = {}
for cd in sorted((CC / "chunks").glob("*_*")):
    c = np.loadtxt(cd / "center.txt"); e = np.loadtxt(cd / "extent.txt")
    names = [im.name for im in read_images_binary(str(cd / "sparse/0/images.bin")).values()]
    chunks[cd.name] = (c, e, names)

fig = plt.figure(figsize=(17, 9.5)); gs = fig.add_gridspec(2, 5, width_ratios=[2.4, 1, 1, 1, 0.08])
ax = fig.add_subplot(gs[:, 0])
cmap = plt.get_cmap("tab20")
ax.scatter(P[:, 0], P[:, 1], c=[cmap(b % 20) for b in B], s=4, linewidths=0)
seams = np.where(np.diff(B) != 0)[0]
ax.scatter(P[seams, 0], P[seams, 1], marker="x", c="k", s=28, linewidths=1.2, label=f"old block hand-off ({len(seams)})", zorder=5)
for k, (c, e, names) in chunks.items():
    ax.add_patch(Rectangle((c[0] - e[0] / 2, c[1] - e[1] / 2), e[0], e[1], fill=False, lw=2, ec="crimson", zorder=4))
    ax.text(c[0] - e[0] / 2 + 0.8, c[1] + e[1] / 2 - 2.2, f"chunk {k}\n{len(names)} cams", color="crimson", fontsize=9, fontweight="bold", zorder=6)
ax.plot(P[0, 0], P[0, 1], "go", ms=9, label="survey start"); ax.plot(P[-1, 0], P[-1, 1], "rs", ms=8, label="survey end")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m, along rows)")
ax.set_title(f"05_13D: {len(order)} keyframes coloured by old lio_row100 block (43 blocks), H3DGS 30 m chunks in red")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
for i, (k, (c, e, names)) in enumerate(chunks.items()):
    a = fig.add_subplot(gs[i // 3, 1 + i % 3])
    a.scatter(P[:, 0], P[:, 1], c="0.85", s=2, linewidths=0)
    S = set(names); pts = np.array([centre[n] for n in names])
    inside = (np.abs(pts[:, 0] - c[0]) <= e[0] / 2) & (np.abs(pts[:, 1] - c[1]) <= e[1] / 2)
    a.scatter(pts[inside, 0], pts[inside, 1], c="tab:blue", s=4, linewidths=0, label=f"inside cell {inside.sum()}")
    a.scatter(pts[~inside, 0], pts[~inside, 1], facecolors="none", edgecolors="tab:orange", s=12, linewidths=0.6, label=f"pulled in by visibility {(~inside).sum()}")
    a.add_patch(Rectangle((c[0] - e[0] / 2, c[1] - e[1] / 2), e[0], e[1], fill=False, lw=1.5, ec="crimson"))
    a.set_aspect("equal"); a.set_title(f"chunk {k} training cameras", fontsize=10); a.legend(fontsize=7, loc="lower right"); a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig("/home/paperspace/logs/h3dgs_topdown.png", dpi=110)
tot_in = sum(int(((np.abs(np.array([centre[n] for n in names])[:, 0] - c[0]) <= e[0] / 2) & (np.abs(np.array([centre[n] for n in names])[:, 1] - c[1]) <= e[1] / 2)).sum()) for c, e, names in chunks.values())
print(f"cells {len(chunks)}; camera slots {sum(len(v[2]) for v in chunks.values())} of which inside-cell {tot_in}, pulled in {sum(len(v[2]) for v in chunks.values()) - tot_in}; old hand-offs {len(seams)}")
