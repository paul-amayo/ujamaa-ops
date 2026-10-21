"""Grow-bag change Dec -> Apr through the STPG: each April bag is carried into the December frame by the
localisation edge of its nearest localised April keyframe (local registration), then matched to the
December bags. Compared with the ledger's global yaw+t registration."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from scipy.spatial import cKDTree
from offline_vtr import monos
K = "/home/paperspace/data/klapmuts"; D = K + "/dec_2025_ten_rows"; A = K + "/apr_2026_zed"
O = D + "/experimental/azalai/stpg_dec_apr_20261021"
L = json.load(open(D + "/experimental/sankofa/klapmuts_ledger_v5.json"))
r = L["registration_A_to_B"]; th = np.radians(r["yaw_deg"]); Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); tt = np.array(r["t"])
A_apr = np.array(L["A_positions_apr_frame"]); B = np.array(L["B_positions"]); A_dec = (A_apr - tt) @ Rz
l2c = sp.load_l2c(D); R = l2c[:3, :3]
to_cam = lambda p2: (R @ np.column_stack([p2[:, 0], p2[:, 1], np.zeros(len(p2))]).T).T   # laser-world ground point -> camera-world 3-D
decC = to_cam(A_dec); aprC = to_cam(B)
g = json.load(open(O + "/lo_superpoint/stpg.json")); dec = np.asarray(g["experiences"]["dec"]["c2w"]); apr = np.asarray(g["experiences"]["apr"]["c2w"])
edges = {s["k"]: (dec[s["j"]] @ np.asarray(s["T_teach_repeat"]), s["n_inliers"]) for s in g["spatial"] if s["T_teach_repeat"] is not None}
MIN_INL = 20
ks = np.array([k for k, (T, n) in edges.items() if n >= MIN_INL]); kp = apr[ks][:, [0, 2], 3]; tree_k = cKDTree(kp)
treeD = cKDTree(decC[:, [0, 2]])
WIN, MIN_K = 5.0, 10
est_xz = np.array([edges[k][0][[0, 2], 3] for k in ks])          # localised April keyframes in the Dec frame
def local_rigid(center_xz):
    """yaw + t (2-D) from April ground positions to Dec-frame estimates, least squares over the localised
    keyframes within WIN m of the bag: the edge noise (~0.15 m each) averages down, lane drift is kept local."""
    idx = np.array(tree_k.query_ball_point(center_xz, WIN))
    if len(idx) < MIN_K: return None
    X, Y = kp[idx], est_xz[idx]; keep = np.ones(len(idx), bool)
    for _ in range(4):                       # trimmed least squares: drop edges > 2x the median residual
        cx, cy = X[keep].mean(0), Y[keep].mean(0)
        U, _, Vt = np.linalg.svd((X[keep] - cx).T @ (Y[keep] - cy)); Rr = Vt.T @ U.T
        if np.linalg.det(Rr) < 0: Rr = Vt.T @ np.diag([1, -1]) @ U.T
        res = np.linalg.norm((X - cx) @ Rr.T + cy - Y, axis=1)
        keep = res <= max(2 * np.median(res[keep]), 0.15)
        if keep.sum() < MIN_K: return None
    return Rr, cy - cx @ Rr.T, int(keep.sum()), float(np.median(res[keep]))
rows = []; fit_res = []
for i, p in enumerate(aprC):
    f = local_rigid(p[[0, 2]])
    if f is None:
        rows.append((i, None, None, "no_edge")); continue
    Rr, t2, nk, med = f; fit_res.append(med)
    q = Rr @ p[[0, 2]] + t2
    dn, jn = treeD.query(q)
    rows.append((i, int(jn), float(dn), "local"))
print(f"windowed rigid fits: keyframes per window median {np.median([local_rigid(p[[0,2]])[2] for p in aprC[::10] if local_rigid(p[[0,2]])])}, fit residual median {np.median(fit_res):.3f} m")
dl = np.array([r[2] for r in rows if r[2] is not None]); n_edge = sum(1 for r in rows if r[3] == "local")
print(f"April bags {len(B)}: carried by a local edge {n_edge} (>= {MIN_INL} inliers within 4 m), no edge {len(B) - n_edge}")
for th_ in (0.3, 0.5, 1.0):
    print(f"  NN Dec bag within {th_} m: {int((dl <= th_).sum())} ({100 * (dl <= th_).mean():.0f} % of carried)")
print(f"  NN distance median {np.median(dl):.2f} m, 75th {np.percentile(dl, 75):.2f}, 90th {np.percentile(dl, 90):.2f}")
# the global ledger registration for the same bags (its own frames, 2-D)
dg, _ = cKDTree(A_apr).query(B)
print(f"ledger global registration: NN within 0.3 m {int((dg <= 0.3).sum())}, 0.5 m {int((dg <= 0.5).sum())} (ledger reports {L['association']['n_pairs']} mutual pairs), median {np.median(dg):.2f} m")
# mutual matching on the local result: April bag i -> Dec bag j within 0.5 m and j's nearest carried April bag is i
carried = [(r[0], r[1], r[2]) for r in rows if r[3] == "local"]
qpts = {}
for i, j, dn in carried: qpts[i] = j
apr_near_dec = {}
for i, j, dn in carried:
    if dn <= 0.5 and (j not in apr_near_dec or dn < apr_near_dec[j][1]): apr_near_dec[j] = (i, dn)
pairs = [(i, j, dn) for i, j, dn in carried if dn <= 0.5 and apr_near_dec.get(j, (None,))[0] == i]
moved = [(i, j, dn) for i, j, dn in pairs if dn > 0.25]
unmatched_apr = [i for i, j, dn in carried if not any(i == p[0] for p in pairs)]
matched_dec = {j for _, j, _ in pairs}
dec_seen = set()   # Dec bags within reach of any carried April keyframe (so "removed" is not "unobserved")
tree_apr_k = cKDTree(np.array([(edges[k][0])[[0, 2], 3] for k in ks]))
for j, p in enumerate(decC):
    if tree_apr_k.query(p[[0, 2]])[0] <= 4.0: dec_seen.add(j)
removed = [j for j in dec_seen if j not in matched_dec]
print(f"local mutual pairs <= 0.5 m: {len(pairs)}; of which moved 0.25-0.5 m: {len(moved)}; April bags without a Dec partner (added or moved > 0.5 m): {len(unmatched_apr)}; Dec bags in reach with no April partner (removed or moved > 0.5 m): {len(removed)} of {len(dec_seen)} in reach")
np.savez(O + "/bag_change_local.npz", apr_idx=[p[0] for p in pairs], dec_idx=[p[1] for p in pairs], dist=[p[2] for p in pairs],
         unmatched_apr=unmatched_apr, removed_dec=removed, decC=decC, aprC=aprC, carried=np.array([(i, j, dn) for i, j, dn in carried]))
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 11))
ok = ~np.isnan(dec[:, 0, 0]); ax.plot(dec[ok][:, 0, 3], dec[ok][:, 2, 3], "-", color="#b8c0c8", lw=1, label="December keyframe path")
ax.scatter(decC[:, 0], decC[:, 2], s=14, c="#7f8c8d", label=f"December bags ({len(decC)})")
Q = np.array([(lambda f: (f[0] @ p[[0, 2]] + f[1]) if f else p[[0, 2]])(local_rigid(p[[0, 2]])) for p in aprC])
cls = np.array(["carried"] * len(aprC), dtype=object)
for i, j, dn in pairs: cls[i] = "moved" if dn > 0.25 else "same"
for i in unmatched_apr: cls[i] = "added?"
for r_ in rows:
    if r_[3] == "no_edge": cls[r_[0]] = "no edge"
for c, col in [("same", "#27ae60"), ("moved", "#f39c12"), ("added?", "#e74c3c"), ("no edge", "#95a5a6")]:
    m = cls == c
    if m.any(): ax.scatter(Q[m, 0], Q[m, 1], s=12, c=col, label=f"April bags in Dec frame: {c} ({m.sum()})")
rm = np.array(removed, dtype=int)
if len(rm): ax.scatter(decC[rm, 0], decC[rm, 2], s=60, facecolors="none", edgecolors="#c0392b", label=f"Dec bags in reach with no April partner ({len(rm)})")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=8); ax.set_title("Grow bags, December vs April, registered through windowed rigid fits of the STPG edges (>= 20 inliers, 8 m windows)")
plt.tight_layout(); plt.savefig(O + "/bag_change_local.png", dpi=110); print("saved", O + "/bag_change_local.png")
