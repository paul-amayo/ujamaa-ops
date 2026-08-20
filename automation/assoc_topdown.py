"""Top-down: 01/02/03 trajectories + offset-free tree association (01<->03)."""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from gps_from_mono import read_gnss_mono, read_kf_mono_timestamps, normalize_ms

DATA = "/home/paperspace/data/citrus_all"
R_E = 6378137.0
A = np.load(f"{DATA}/sankofa_substrate/tree_assoc_abs_01_03.npz")
LAT0, LON0 = A["origin"]
i1, e1 = A["id_a"], A["e_a"]
i3, e3 = A["id_b"], A["e_b"]
pairs = A["pairs"]
pitch = float(A["tree_pitch"]); shift = A["residual_shift"]

# trajectories in the SAME frame as the association's origin
tracks = {}
for s, tag in [("01_13B_Jackal", "01"), ("02_13B_Jackal", "02"), ("03_13B_Jackal", "03")]:
    gts, glat, glon, *_ = read_gnss_mono(f"{DATA}/{s}/gps.monolithic")
    kts = read_kf_mono_timestamps(f"{DATA}/{s}/image_left_kf20cm.monolithic")
    gms, kms = normalize_ms(gts), normalize_ms(kts)
    o = np.argsort(gms); gs = gms[o]
    i = np.clip(np.searchsorted(gs, kms), 1, len(gs) - 1)
    nn = np.where(np.abs(kms - gs[i - 1]) <= np.abs(kms - gs[i]), i - 1, i)
    la, lo = glat[o[nn]], glon[o[nn]]
    tracks[tag] = np.column_stack([
        np.radians(lo - LON0) * R_E * np.cos(np.radians(LAT0)),
        np.radians(la - LAT0) * R_E])

p1 = {int(a) for a, b, _ in pairs}
p3 = {int(b) for a, b, _ in pairs}
m1 = np.array([int(i) in p1 for i in i1])
m3 = np.array([int(i) in p3 for i in i3])
dd = pairs[:, 2].astype(float)
print(f"pairs {len(pairs)}, median {np.median(dd):.2f} m | 01 unmatched {(~m1).sum()}, "
      f"03 unmatched {(~m3).sum()}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUT, GRID = "#1a1a19", "#6f6e66", "#e6e5dd"
C = {"01": "#2a78d6", "02": "#eb6834", "03": "#1baf7a"}
fig, (ax, axz) = plt.subplots(2, 1, figsize=(15, 9.5), dpi=140,
                              gridspec_kw={"height_ratios": [1.5, 1]})
fig.patch.set_facecolor("#fff")
for a_ in (ax, axz):
    a_.set_facecolor("#fff"); a_.grid(True, color=GRID, lw=.6); a_.set_axisbelow(True)
    for sp in a_.spines.values(): sp.set_visible(False)
    a_.tick_params(colors=MUT, labelsize=8); a_.set_aspect("equal")

for tag in ["01", "02", "03"]:
    t = tracks[tag]
    ax.plot(t[:, 0], t[:, 1], "-", lw=.5, color=C[tag], alpha=.30)
ax.plot([], [], "-", lw=1.5, color=C["01"], alpha=.6, label="01 track")
ax.plot([], [], "-", lw=1.5, color=C["02"], alpha=.6, label="02 track (sparse, 9 legs)")
ax.plot([], [], "-", lw=1.5, color=C["03"], alpha=.6, label="03 track")
for a, b, _ in pairs:
    ka = np.where(i1 == int(a))[0][0]; kb = np.where(i3 == int(b))[0][0]
    ax.plot([e1[ka, 0], e3[kb, 0]], [e1[ka, 1], e3[kb, 1]], "-", color="#52514e", lw=.9)
ax.scatter(e1[m1, 0], e1[m1, 1], s=26, facecolors="none", edgecolors=C["01"], lw=1.2,
           label=f"01 tree, matched ({m1.sum()})")
ax.scatter(e3[m3, 0], e3[m3, 1], s=26, marker="s", facecolors="none",
           edgecolors=C["03"], lw=1.2, label=f"03 tree, matched ({m3.sum()})")
ax.scatter(e1[~m1, 0], e1[~m1, 1], s=34, marker="x", color="#d03b3b", lw=1.3,
           label=f"unmatched ({(~m1).sum()} of 01, {(~m3).sum()} of 03)")
ax.scatter(e3[~m3, 0], e3[~m3, 1], s=34, marker="x", color="#d03b3b", lw=1.3)
ax.legend(loc="upper center", ncol=6, frameon=False, fontsize=8.5, labelcolor=INK,
          bbox_to_anchor=(.5, 1.14))
ax.set_title("13B — trajectories + offset-free tree association "
             f"({len(pairs)} pairs, median {np.median(dd):.2f} m; "
             f"residual shift {np.linalg.norm(shift):.2f} m vs {pitch:.2f} m tree pitch)",
             color=INK, fontsize=12, loc="left")
ax.set_ylabel("North (m)", color=MUT, fontsize=9)

# zoom: 40 m window, links visible against tree pitch
x0 = -20
sel1 = (e1[:, 0] > x0) & (e1[:, 0] < x0 + 45)
sel3 = (e3[:, 0] > x0) & (e3[:, 0] < x0 + 45)
for tag in ["01", "03"]:
    t = tracks[tag]
    k = (t[:, 0] > x0) & (t[:, 0] < x0 + 45)
    axz.plot(t[k, 0], t[k, 1], "-", lw=.6, color=C[tag], alpha=.35)
for a, b, d_ in pairs:
    ka = np.where(i1 == int(a))[0][0]; kb = np.where(i3 == int(b))[0][0]
    if x0 < e1[ka, 0] < x0 + 45:
        axz.plot([e1[ka, 0], e3[kb, 0]], [e1[ka, 1], e3[kb, 1]], "-", color="#52514e", lw=1.4)
axz.scatter(e1[sel1, 0], e1[sel1, 1], s=60, facecolors="none", edgecolors=C["01"], lw=1.4)
axz.scatter(e3[sel3, 0], e3[sel3, 1], s=60, marker="s", facecolors="none",
            edgecolors=C["03"], lw=1.4)
axz.set_title(f"zoom (45 m window): each link is one canonical tree — links are short "
              f"({np.median(dd):.2f} m) relative to the {pitch:.2f} m spacing between trees",
              color=INK, fontsize=11, loc="left")
axz.set_xlabel("East (m)", color=MUT, fontsize=9); axz.set_ylabel("North (m)", color=MUT, fontsize=9)
fig.tight_layout()
out = "/home/paperspace/code/lab_notebook/figs/assoc_abs_topdown.png"
fig.savefig(out, bbox_inches="tight")
print("figure ->", out)
