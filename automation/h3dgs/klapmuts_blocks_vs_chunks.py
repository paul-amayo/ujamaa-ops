#!/usr/bin/env python
# How the lio_row100 blocks (trajectory segments) map onto the H3DGS chunks (30 m spatial cells) for
# Klapmuts apr_2026_zed. Prints a block x chunk keyframe table and draws one panel per chunk showing which
# cameras train it (inside the cell vs pulled in from outside by visibility), coloured by block.
# Run: ~/miniconda3/envs/h3dgs/bin/python klapmuts_blocks_vs_chunks.py  -> ~/logs/klapmuts_blocks_vs_chunks.png
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
import sys as _s
S = Path(_s.argv[1] if len(_s.argv) > 1 else "/home/paperspace/data/klapmuts/apr_2026_zed"); P = S / (_s.argv[2] if len(_s.argv) > 2 else "experimental/h3dgs") / "camera_calibration"; BL = S / "prod/tassili/blocks_ns/lio_row100"
OUT = _s.argv[3] if len(_s.argv) > 3 else "/home/paperspace/logs/klapmuts_blocks_vs_chunks.png"
cen = lambda ims: {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()}
prior = cen(read_images_binary(str(P / "prior/sparse/0/images.bin")))
order, block_of = [], {}
for tj in sorted(x for x in BL.glob("block_[0-9][0-9][0-9]/transforms.json") if x.parent.name[6:].isdigit()):
    b = int(tj.parent.name[6:])
    for f in json.load(open(tj))["frames"]:
        n = Path(f["file_path"]).name; order.append(n); block_of[n] = b
X = np.array([prior[n] for n in order]); B = np.array([block_of[n] for n in order]); blocks = sorted(set(B))
chunks = {}
for cd in sorted((P / "chunks").glob("*_*")):
    c = np.loadtxt(cd / "center.txt"); e = np.loadtxt(cd / "extent.txt")
    trained = set(im.name for im in read_images_binary(str(cd / "sparse/0/images.bin")).values())
    inside = (np.abs(X[:, 0] - c[0]) <= e[0] / 2) & (np.abs(X[:, 1] - c[1]) <= e[1] / 2)
    chunks[cd.name] = dict(c=c, e=e, trained=np.array([n in trained for n in order]), inside=inside)
names = list(chunks)
print("block  kf  " + "  ".join(f"{n:>9}" for n in names) + "   (keyframes inside each cell | used to train it)")
for b in blocks:
    m = B == b; row = "  ".join(f"{int((chunks[n]['inside'] & m).sum()):>4}|{int((chunks[n]['trained'] & m).sum()):<4}" for n in names)
    spans = sum(1 for n in names if (chunks[n]["inside"] & m).any()); feeds = sum(1 for n in names if (chunks[n]["trained"] & m).any())
    print(f"  {b:03d} {int(m.sum()):4d}  {row}   cells spanned {spans}, chunks fed {feeds}")
print("chunk  inside  trained  extra-from-outside  blocks-inside  blocks-feeding")
for n in names:
    ch = chunks[n]; print(f"  {n}  {int(ch['inside'].sum()):5d}  {int(ch['trained'].sum()):6d}  {int((ch['trained'] & ~ch['inside']).sum()):6d}  {len(set(B[ch['inside']])):6d}  {len(set(B[ch['trained']])):6d}")
print(f"keyframes {len(order)}, blocks {len(blocks)}, in no cell: {int((~np.any([chunks[n]['inside'] for n in names], axis=0)).sum())}, in no training set: {int((~np.any([chunks[n]['trained'] for n in names], axis=0)).sum())}")
cmap = plt.get_cmap("tab20"); fig, axs = plt.subplots(2, 2, figsize=(15, 13))
for a, n in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], ["0_1", "1_1", "0_0", "1_0"]):
    ch = chunks[n]; a.scatter(X[:, 0], X[:, 1], c="0.88", s=4, linewidths=0, label="all keyframes")
    ex = ch["trained"] & ~ch["inside"]; a.scatter(X[ex, 0], X[ex, 1], c=[cmap(b % 20) for b in B[ex]], s=14, marker="^", linewidths=0, label="outside the cell, pulled in by visibility")
    a.scatter(X[ch["inside"], 0], X[ch["inside"], 1], c=[cmap(b % 20) for b in B[ch["inside"]]], s=9, linewidths=0, label="inside the cell")
    a.add_patch(Rectangle((ch["c"][0] - ch["e"][0] / 2, ch["c"][1] - ch["e"][1] / 2), ch["e"][0], ch["e"][1], fill=False, ec="crimson", lw=2))
    for b in sorted(set(B[ch["trained"]])):
        m = ch["trained"] & (B == b); a.text(*X[m].mean(0)[:2], f"{b:02d}", fontsize=7, ha="center", va="center", color="k", bbox=dict(fc="w", ec="none", alpha=.7, pad=0.4))
    a.set_aspect("equal"); a.grid(alpha=.25); a.set_title(f"chunk {n}: {int(ch['inside'].sum())} keyframes inside, {int(ch['trained'].sum())} used for training, from {len(set(B[ch['trained']]))} blocks")
    a.legend(loc="lower right", fontsize=8)
fig.suptitle(f"{S.name} — which blocks feed which 30 m chunk (block ids at each block's centre; colours = blocks)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT, dpi=100); print("wrote", OUT)
