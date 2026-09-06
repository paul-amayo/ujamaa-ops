#!/usr/bin/env python3
"""Near-field, plant-band top-down of the ten_rows cloud + row-periodicity check.
Ground = p2 of height (-y); band = ground+0.15 .. ground+3.0 m; within R m (ground
plane) of the camera path. Rows run along the path legs, so we histogram the
coordinate PERPENDICULAR to the path's principal direction to count rows."""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--R", type=float, default=12.0); ap.add_argument("--cell", type=float, default=0.10)
a = ap.parse_args()
z = np.load(a.npz); xyz, cnt, cams = z["xyz"], z["count"], z["cams"]
h = -xyz[:, 1]; ground = float(np.percentile(h, 2))
band = (h > ground + 0.15) & (h < ground + 1.8)
d2, _ = cKDTree(cams[:, [0, 2]]).query(xyz[:, [0, 2]], workers=8)
sel = band & (d2 <= a.R)
P = xyz[sel]; Hs = h[sel] - ground; C = cnt[sel]
print(f"[v2] ground {ground:.2f} m below sensor; voxels in band & within {a.R} m of path: {sel.sum():,} of {len(xyz):,}")

# row periodicity: PCA of path -> along-row e1, perpendicular e2
pc = cams[:, [0, 2]] - cams[:, [0, 2]].mean(0)
e1 = np.linalg.svd(pc, full_matrices=False)[2][0]; e2 = np.array([-e1[1], e1[0]])
u = (P[:, [0, 2]] - cams[:, [0, 2]].mean(0)) @ e2
uc = pc @ e2
lo, hi = uc.min() - 3, uc.max() + 3
bins = np.arange(lo, hi, 0.25); hist, _ = np.histogram(u, bins=bins, weights=C)
pk, pr = find_peaks(hist, distance=int(1.5 / 0.25), prominence=hist.max() * 0.05)
pkx = bins[pk] + 0.125
print(f"[v2] path principal dir e1={np.round(e1,3)}; perpendicular row peaks: {len(pk)} at u={np.round(pkx,1)} m")
if len(pkx) > 1: print(f"[v2] row spacing median {np.median(np.diff(pkx)):.2f} m (range {np.diff(pkx).min():.2f}..{np.diff(pkx).max():.2f})")

x, y = P[:, 0], P[:, 2]; x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
nx, ny = int((x1 - x0) / a.cell) + 1, int((y1 - y0) / a.cell) + 1
ix = np.clip(((x - x0) / a.cell).astype(int), 0, nx - 1); iy = np.clip(((y - y0) / a.cell).astype(int), 0, ny - 1)
flat = ix * ny + iy
Hsum = np.zeros(nx * ny); Hn = np.zeros(nx * ny); np.add.at(Hsum, flat, Hs); np.add.at(Hn, flat, 1); Hmax = np.where(Hn > 0, Hsum / np.maximum(Hn, 1), np.nan)
D = np.zeros(nx * ny); np.add.at(D, flat, C)
Hm = Hmax.reshape(nx, ny).T; Dm = np.log1p(D.reshape(nx, ny).T); Dm[D.reshape(nx, ny).T == 0] = np.nan

fig, axs = plt.subplots(1, 3, figsize=(24, 9), dpi=120, gridspec_kw={"width_ratios": [1, 1, 0.6]})
fig.patch.set_facecolor("#1d1a17")
for ax, img, cmap, vmin, vmax, ttl in [
    (axs[0], Hm, "viridis", 0, 1.5, f"plant-band MEAN height (m above ground), within {a.R:.0f} m of path"),
    (axs[1], Dm, "magma", None, None, "return density (log count) in plant band")]:
    ax.set_facecolor("#111"); im = ax.imshow(img, origin="lower", extent=[x0, x1, y0, y1], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.plot(cams[:, 0], cams[:, 2], color="#ff5533", lw=0.8, alpha=0.9)
    ax.set_aspect("equal"); ax.set_title(ttl, color="#e8e0d4", fontsize=11); ax.tick_params(colors="#ccc")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.ax.yaxis.set_tick_params(color="#ccc"); plt.setp(cb.ax.get_yticklabels(), color="#ccc")
axs[2].set_facecolor("#111"); axs[2].plot(bins[:-1] + 0.125, hist, color="#7fd", lw=1.2)
axs[2].plot(pkx, hist[pk], "o", color="#ff5533", ms=5)
for xv in uc[::50]: axs[2].axvline(xv, color="#ff5533", alpha=0.08, lw=0.6)
axs[2].set_title(f"returns vs distance PERPENDICULAR to rows — {len(pk)} peaks", color="#e8e0d4", fontsize=11)
axs[2].set_xlabel("u (m, perpendicular to path legs)", color="#ccc"); axs[2].tick_params(colors="#ccc")
fig.suptitle(f"dec_2025_ten_rows — plant band 0.15..1.8 m above ground, ground {ground:.2f} m below sensor", color="#e8e0d4", fontsize=13)
fig.tight_layout(); fig.savefig(a.out, facecolor=fig.get_facecolor()); print(f"[v2] wrote {a.out}")
