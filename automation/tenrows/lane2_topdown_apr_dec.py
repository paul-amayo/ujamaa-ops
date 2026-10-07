"""Top-down of Klapmuts lane 2, April vs December, in one frame (Paul 2026-10-06: "train but on the right lane. Show the top down
of lane 2 April and December before you train"). Common frame = the plant ledger's April frame ((y, -x) of the April H3DGS export
frame); December enters through the ledger v5 registration (p_apr = R(1.2428 deg) p_dec + (0.1138, 1.5920), December LO world),
the April lane through a Sim(3) of its lane-LO cameras onto the same frames' export poses. Panels: (a) the site - both surveys'
camera tracks, both plant ledgers' grow bags, the two lane-2 training tracks; (b) April lane-2 LiDAR init from above, (c) December
lane-2 LiDAR init from above, same axes, both tracks on both, points between ground and 2.2 m coloured by height.
  h3dgs env: lane2_topdown_apr_dec.py [april lane dir] [out png] -> data/demo_video_v2/klapmuts_compare/lane2_topdown_apr_vs_dec.png
v2 (Paul: "only for lane 2"): lane panels only, the lane laid horizontally (along = ledger y), April above December; the April
track is the placed training cameras (transforms_ref_lo.json) when they exist, else the LO base (before the SfM)."""
import json, re, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
K = "/home/paperspace/data/klapmuts"; AL = sys.argv[1] if len(sys.argv) > 1 else f"{K}/apr_2026_zed/experimental/lane2_apr"; DL = f"{K}/dec_2025_ten_rows/experimental/lane2"; GL = np.diag([1.0, -1, -1, 1])
L5 = json.load(open(f"{K}/dec_2025_ten_rows/experimental/sankofa/klapmuts_ledger_v5.json")); yaw = np.radians(L5["registration_A_to_B"]["yaw_deg"]); tt = np.array(L5["registration_A_to_B"]["t"])
Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
dec2l = lambda P: P[:, :2] @ Rz.T + tt                     # December LO world -> ledger April frame
exp2l = lambda P: np.stack([P[:, 1], -P[:, 0]], 1)         # April export frame -> ledger April frame
def tf(p):
    J = json.load(open(p)); return {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"], float) @ GL for f in J["frames"]}
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T; s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X); return s, R, my - s * R @ mx
def ply(p):
    raw = open(p, "rb").read(); h = raw.index(b"end_header\n") + 11; n = int(re.search(rb"element vertex (\d+)", raw[:h]).group(1))
    r = np.frombuffer(raw[h:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n); return np.c_[r["x"], r["y"], r["z"]].astype(np.float64)
# April lane: training cameras (warped, lane-LO frame) -> export frame by Sim(3) on the same frames' export poses
import os; AWJ = "transforms_ref_lo.json" if os.path.exists(f"{AL}/transforms_ref_lo.json") else "transforms_lo.json"
aw, alo, aex = tf(f"{AL}/{AWJ}"), tf(f"{AL}/transforms_lo.json"), tf(f"{AL}/transforms_lo_export.json")
an = sorted(aw, key=lambda n: int(re.sub(r"\D", "", n))); s, R, t = umeyama(np.array([alo[n][:3, 3] for n in an]), np.array([aex[n][:3, 3] for n in an]))
lo2exp = lambda P: s * P @ R.T + t
A_trk = exp2l(lo2exp(np.array([aw[n][:3, 3] for n in an]))); A_pts = ply(f"{AL}/init_lidar.ply"); A_xy = exp2l(lo2exp(A_pts)); A_z = lo2exp(A_pts)[:, 2]
A_camz = lo2exp(np.array([aw[n][:3, 3] for n in an]))[:, 2]
# December lane 2: training cameras + init (LO world)
dw = tf(f"{DL}/transforms_ref_lo.json"); dn = sorted(dw, key=lambda n: int(re.sub(r"\D", "", n))); D_trk = dec2l(np.array([dw[n][:3, 3] for n in dn]))
D_pts = ply(f"{DL}/init_lidar.ply"); D_xy = dec2l(D_pts); D_z = D_pts[:, 2]; D_camz = np.array([dw[n][:3, 3] for n in dn])[:, 2]
# whole surveys: April export keyframes, December LO track (camera = laser pose x laser->camera^-1)
ims = read_images_binary(f"{K}/apr_2026_zed/experimental/h3dgs/camera_calibration/aligned/sparse/0/images.bin")
A_all = exp2l(np.array([-qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()]))
z = np.load(f"{K}/dec_2025_ten_rows/experimental/laser_dump/lo_poses.npz"); C2L = np.linalg.inv(np.array(json.load(open(f"{K}/dec_2025_ten_rows/prod/monos/rig.json"))["laser_to_camera_left"], float))
D_all = dec2l((z["T"] @ C2L)[:, :3, 3])
Ab, Db = np.array(L5["B_positions"]), np.array(L5["A_positions_apr_frame"])
# checks: lateral distance between the two lane tracks; each survey's lane cameras to its own nearest bag
from scipy.spatial import cKDTree
lat = cKDTree(D_trk).query(A_trk)[0]; ab = cKDTree(Ab).query(A_trk)[0]; db = cKDTree(Db).query(D_trk)[0]
ov = (A_trk[:, 1] >= D_trk[:, 1].min()) & (A_trk[:, 1] <= D_trk[:, 1].max())
msg = (f"April lane-2 track to December lane-2 track (where both run, {int(ov.sum())} April frames): median {np.median(lat[ov]):.2f} m, p90 {np.percentile(lat[ov], 90):.2f} m; "
       f"camera to nearest own-survey grow bag: April {np.median(ab):.2f} m, December {np.median(db):.2f} m; April lane LO->export Sim(3) scale {s:.4f}")
print("[topdown]", msg, flush=True)
def band(xy, zz, camz):   # ground .. 2.2 m above it (the ground = 5th percentile of z within 2 m of the lane axis)
    g = np.percentile(zz[np.abs(xy[:, 0] - np.median(xy[:, 0])) < 2.0], 5); m = (zz > g - 0.3) & (zz < g + 2.2); return m, g
x0 = np.median(np.r_[A_trk[:, 0], D_trk[:, 0]]); y0, y1 = min(A_trk[:, 1].min(), D_trk[:, 1].min()) - 2, max(A_trk[:, 1].max(), D_trk[:, 1].max()) + 2
fig, axs = plt.subplots(2, 1, figsize=(18, 8.6), sharex=True, gridspec_kw=dict(hspace=0.12))
for a2, (lab, xy, zz, camz) in zip(axs, ((f"April lane 2 (LiDAR from above; {len(an)} lane frames)", A_xy, A_z, A_camz), (f"December lane 2 (LiDAR from above; {len(dn)} lane frames)", D_xy, D_z, D_camz))):
    m, g = band(xy, zz, camz); w = m & (np.abs(xy[:, 0] - x0) < 6) & (xy[:, 1] > y0) & (xy[:, 1] < y1); o = np.argsort(zz[w])
    sc = a2.scatter(xy[w][o, 1], xy[w][o, 0], s=0.12, c=(zz[w][o] - g), cmap="turbo", vmin=0, vmax=2.0, rasterized=True)
    a2.plot(D_trk[:, 1], D_trk[:, 0], lw=2.4, c="#2166ac", label="December lane-2 track"); a2.plot(A_trk[:, 1], A_trk[:, 0], lw=1.4, c="#e66101", ls="--", label="April lane-2 track")
    a2.set_xlim(y0, y1); a2.set_ylim(x0 - 6, x0 + 6); a2.set_aspect("equal"); a2.set_title(lab, fontsize=12, loc="left"); a2.grid(alpha=0.25); a2.set_ylabel("across (m)")
axs[0].legend(loc="upper right", fontsize=9, framealpha=0.9); axs[1].set_xlabel("along the lane (m, ledger frame)")
cb = fig.colorbar(sc, ax=axs, fraction=0.012, pad=0.01); cb.set_label("height above ground (m)")
fig.suptitle("Klapmuts lane 2, April vs December, same frame (plant-ledger registration). " + msg.replace("; camera", ";\ncamera"), fontsize=11, y=1.0)
out = sys.argv[2] if len(sys.argv) > 2 else "/home/paperspace/data/demo_video_v2/klapmuts_compare/lane2_topdown_apr_vs_dec.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print("[topdown] ->", out, flush=True)
