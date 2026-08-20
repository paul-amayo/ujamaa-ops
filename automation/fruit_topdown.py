"""Top-down of 04 fruit counts on tree positions (all-views pass)."""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from gps_from_mono import read_gnss_mono, normalize_ms

DATA = "/home/paperspace/data/citrus_all"
S = "04_13D_Jackal"
R_E = 6378137.0
F = json.load(open(f"{DATA}/{S}/sam3_v2/fruit_v1_allviews.json"))["trees"]
G = json.load(open(f"{DATA}/{S}/sam3_v2/global_ids.json"))["stats"]

# place trees in the survey's own absolute frame (04 gps is true WGS84)
gts, gla, glo, *_ = read_gnss_mono(f"{DATA}/{S}/gps.monolithic")
g = normalize_ms(gts); o = np.argsort(g); gs, gla, glo = g[o], gla[o], glo[o]
lat0, lon0 = gla.mean(), glo.mean()
gen = np.column_stack([np.radians(glo - lon0) * R_E * np.cos(np.radians(lat0)),
                       np.radians(gla - lat0) * R_E])
lio = json.load(open(f"{DATA}/{S}/lio_image_poses_kf20cm.json"))
lts = normalize_ms(np.array([p["timestamp_ns"] for p in lio], float))
lat = json.load(open(f"{DATA}/sankofa_substrate/gps_lio_latency.json")).get(S, {})
lts = lts + float(lat.get("best_dt_ms", 0.0))
i = np.clip(np.searchsorted(gs, lts), 1, len(gs) - 1)
nn = np.where(np.abs(lts - gs[i - 1]) <= np.abs(lts - gs[i]), i - 1, i)
keep = np.abs(gs[nn] - lts) < 100.0
xz = np.array([[np.asarray(p["transform"])[0, 3],
                np.asarray(p["transform"])[2, 3]] for p in lio])[keep]
dst = gen[nn][keep]
cs, cd = xz.mean(0), dst.mean(0)
H = (xz - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1] *= -1; R = Vt.T @ U.T
t = cd - R @ cs

gid, pos, fmax, ftot, nviews = [], [], [], [], []
for k, st in G.items():
    if k not in F:
        continue
    c = np.array([st["world_centroid"][0], st["world_centroid"][2]])
    gid.append(int(k)); pos.append(c @ R.T + t)
    fmax.append(F[k]["fruit_count_max"]); ftot.append(sum(F[k]["per_view_counts"]))
    nviews.append(F[k]["n_views"])
pos = np.array(pos); fmax = np.array(fmax); ftot = np.array(ftot); nviews = np.array(nviews)
traj = gen
print(f"{len(gid)} trees placed | fruiting {(fmax>0).sum()} | max {fmax.max()}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUT, GRID = "#1a1a19", "#6f6e66", "#e6e5dd"
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=140,
                              gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor("#fff")
for a in (ax, ax2):
    a.set_facecolor("#fff"); a.grid(True, color=GRID, lw=.6); a.set_axisbelow(True)
    for sp in a.spines.values(): sp.set_visible(False)
    a.tick_params(colors=MUT, labelsize=8); a.set_aspect("equal")

ax.plot(traj[:, 0], traj[:, 1], "-", lw=.5, color="#b9c4cf", alpha=.9, label="04 track")
z = fmax == 0
ax.scatter(pos[z, 0], pos[z, 1], s=30, facecolors="none", edgecolors="#b4b2a9", lw=1.0,
           label=f"no fruit detected ({z.sum()})")
nz = ~z
sc = ax.scatter(pos[nz, 0], pos[nz, 1], s=60 + 34 * fmax[nz], c=fmax[nz],
                cmap="YlOrRd", vmin=1, vmax=fmax.max(), lw=.6, edgecolors="#7a3b12",
                zorder=3, label=f"fruiting ({nz.sum()})")
for k in np.where(nz)[0]:
    ax.annotate(str(fmax[k]), (pos[k, 0], pos[k, 1]), fontsize=7.5, color=INK,
                xytext=(6, 5), textcoords="offset points", zorder=4)
cb = fig.colorbar(sc, ax=ax, shrink=.72, pad=.02)
cb.set_label("fruit in best view", color=MUT, fontsize=8)
cb.ax.tick_params(colors=MUT, labelsize=7); cb.outline.set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_title(f"04_13D — fruit per tree (all {int(nviews.sum())} views scored)\n"
             f"{nz.sum()}/{len(gid)} trees fruiting, max {fmax.max()} in one view",
             color=INK, fontsize=11, loc="left")
ax.set_xlabel("East (m)", color=MUT, fontsize=9); ax.set_ylabel("North (m)", color=MUT, fontsize=9)

# does detection track how often a tree was seen?
ax2.scatter(nviews[z], fmax[z], s=26, facecolors="none", edgecolors="#b4b2a9", lw=1.0)
ax2.scatter(nviews[nz], fmax[nz], s=40, c=fmax[nz], cmap="YlOrRd", vmin=1,
            vmax=fmax.max(), lw=.5, edgecolors="#7a3b12")
ax2.set_aspect("auto")
ax2.set_xscale("log")
ax2.set_xlabel("views of this tree (log)", color=MUT, fontsize=9)
ax2.set_ylabel("fruit in best view", color=MUT, fontsize=9)
ax2.set_title("detection vs observation count —\nis fruit found only on well-observed trees?",
              color=INK, fontsize=11, loc="left")
fig.tight_layout()
out = "/home/paperspace/code/lab_notebook/figs/fruit_04_topdown.png"
fig.savefig(out, bbox_inches="tight")
print("figure ->", out)
from scipy.stats import spearmanr
r, p = spearmanr(nviews, fmax)
print(f"Spearman(views, fruit_max) = {r:+.2f} (p={p:.3f})")
PY = None
