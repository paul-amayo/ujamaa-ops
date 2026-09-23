#!/usr/bin/env python
# Klapmuts apr pose-correction figure (LIO vs global BA vs per-block refine). Recovered from the
# 2026-09-23 session transcript; originally run inline. Writes ~/logs/klapmuts_poses_topdown.png and
# klapmuts_glomap_global.png. Run with the h3dgs env: ~/miniconda3/envs/h3dgs/bin/python klapmuts_pose_figure.py
# Inputs: the H3DGS project's prior / aligned (global-BA) / glomap sparse models + chunk cells, and the
# canonical blocks' transforms.json (refined blocks carry pose_convention containing "COLMAP").
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
S = Path("/home/paperspace/data/klapmuts/apr_2026_zed"); P = S / "experimental/h3dgs/camera_calibration"; BL = S / "prod/tassili/blocks_ns/lio_row100"
R_W = np.array(json.load(open(S / "experimental/h3dgs/export_meta.json"))["world_rotation_to_zup"]); GL2CV = np.diag([1., -1, -1, 1])
cen = lambda ims: {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()}
prior = cen(read_images_binary(str(P / "prior/sparse/0/images.bin"))); ba = cen(read_images_binary(str(P / "aligned/sparse/0/images.bin")))
glo = cen(read_images_binary(str(P / "glomap/sparse/0/images.bin")))
names = sorted(prior); order, block_of, refined_blocks, ref = [], {}, set(), {}
for tj in sorted(x for x in BL.glob("block_[0-9][0-9][0-9]/transforms.json") if x.parent.name[6:].isdigit()):
    t = json.load(open(tj)); b = int(tj.parent.name[6:]); isref = "COLMAP" in str(t.get("pose_convention", ""))
    if isref: refined_blocks.add(b)
    for f in t["frames"]:
        n = Path(f["file_path"]).name; order.append(n); block_of[n] = b
        if isref: ref[n] = (R_W @ np.array(f["transform_matrix"]) @ GL2CV)[:3, 3]
Xp = np.array([prior[n] for n in order]); Xb = np.array([ba[n] for n in order]); B = np.array([block_of[n] for n in order])
dba = np.linalg.norm(Xb - Xp, axis=1); has_ref = np.array([n in ref for n in order]); Xr = np.array([ref[n] if n in ref else prior[n] for n in order]); dref = np.linalg.norm(Xr - Xp, axis=1)
fig, ax = plt.subplots(2, 2, figsize=(17, 13)); cmap = plt.get_cmap("tab20")
a = ax[0, 0]; a.scatter(Xp[:, 0], Xp[:, 1], c=[cmap(b % 20) for b in B], s=5, linewidths=0); seams = np.where(np.diff(B) != 0)[0]; a.scatter(Xp[seams, 0], Xp[seams, 1], marker="x", c="k", s=25)
for cd in sorted((P / "chunks").glob("*_*")):
    c = np.loadtxt(cd / "center.txt"); e = np.loadtxt(cd / "extent.txt"); a.add_patch(Rectangle((c[0] - e[0] / 2, c[1] - e[1] / 2), e[0], e[1], fill=False, ec="crimson", lw=2)); a.text(c[0] - e[0] / 2 + 0.5, c[1] + e[1] / 2 - 2, f"chunk {cd.name}", color="crimson", fontsize=9, fontweight="bold")
a.plot(Xp[0, 0], Xp[0, 1], "go", ms=9); a.plot(Xp[-1, 0], Xp[-1, 1], "rs", ms=8); a.set_aspect("equal"); a.set_title(f"LIO trajectory: {len(order)} keyframes in {len(set(B))} blocks (x = block hand-offs), 30 m chunks"); a.grid(alpha=.25)
a = ax[0, 1]; sc = a.scatter(Xp[:, 0], Xp[:, 1], c=np.minimum(dba * 100, 40), cmap="magma", s=7, linewidths=0); a.quiver(Xp[::3, 0], Xp[::3, 1], (Xb - Xp)[::3, 0], (Xb - Xp)[::3, 1], angles="xy", scale_units="xy", scale=0.25, width=0.002, color="tab:blue", alpha=.6)
plt.colorbar(sc, ax=a, label="|global-BA pose − LIO| (cm, clipped at 40)"); a.set_aspect("equal"); a.set_title(f"where the GLOBAL BA moved the cameras (arrows ×4): median {np.median(dba)*100:.1f} cm, p90 {np.percentile(dba,90)*100:.1f} cm"); a.grid(alpha=.25)
a = ax[1, 0]; m = has_ref; sc = a.scatter(Xp[m, 0], Xp[m, 1], c=np.minimum(dref[m] * 100, 40), cmap="magma", s=7, linewidths=0); a.scatter(Xp[~m, 0], Xp[~m, 1], c="0.85", s=4, linewidths=0)
a.quiver(Xp[m][::3, 0], Xp[m][::3, 1], (Xr - Xp)[m][::3, 0], (Xr - Xp)[m][::3, 1], angles="xy", scale_units="xy", scale=0.25, width=0.002, color="tab:green", alpha=.6)
plt.colorbar(sc, ax=a, label="|per-block refined − LIO| (cm)"); a.set_aspect("equal"); a.set_title(f"where the PER-BLOCK GLOMAP refine moved them ({len(refined_blocks)}/{len(set(B))} blocks done, grey = pending): median {np.median(dref[m])*100:.1f} cm, p90 {np.percentile(dref[m],90)*100:.1f} cm"); a.grid(alpha=.25)
a = ax[1, 1]; idx = np.arange(len(order)); a.plot(idx, dba * 100, lw=.7, color="tab:blue", label="global BA − LIO"); a.plot(idx[m], dref[m] * 100, ".", ms=2, color="tab:green", label="per-block refine − LIO")
for s_ in seams: a.axvline(s_, color="k", alpha=.12, lw=.6)
a.set_ylim(0, 60); a.set_xlabel("keyframe index along the survey (block boundaries in grey)"); a.set_ylabel("displacement from the LIO pose (cm)"); a.legend(); a.set_title("per-keyframe displacement along the trajectory"); a.grid(alpha=.25)
fig.suptitle("Klapmuts apr_2026_zed — pose corrections, z-up survey frame (metres)", fontsize=13); fig.tight_layout(); fig.savefig("/home/paperspace/logs/klapmuts_poses_topdown.png", dpi=105)
# the discarded global GLOMAP, shown honestly at its own (broken) scale
fig2, a = plt.subplots(figsize=(7, 5)); Xg = np.array([glo[n] for n in order]); a.scatter(Xg[:, 0], Xg[:, 1], c=[cmap(b % 20) for b in B], s=4, linewidths=0); a.set_aspect("equal"); a.set_title("discarded: global GLOMAP over all 2006 images (its own units) — rows collapsed"); a.grid(alpha=.25); fig2.tight_layout(); fig2.savefig("/home/paperspace/logs/klapmuts_glomap_global.png", dpi=100)
print(f"blocks {len(set(B))}, refined so far {len(refined_blocks)}; BA move median {np.median(dba)*100:.1f} cm; refine move median {np.median(dref[m])*100:.1f} cm; BA-vs-refine on refined kf: median {np.median(np.linalg.norm(Xb[m]-Xr[m],axis=1))*100:.1f} cm")
