#!/usr/bin/env python3
"""July-style row solve: mv10 census centroids -> HighInterface ransac_init."""
import sys
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib")
import json
import numpy as np
import aru_nerf_interface as a
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

g = json.load(open('/home/paperspace/data/klapmuts/apr_2026_zed/prod/bateleur/sam3_mv10/global_ids.json'))
rows_stats = g['stats'] if isinstance(g['stats'], list) else list(g['stats'].values())
pts = np.array([r['world_centroid'] for r in rows_stats], dtype=np.float64)
xz = pts[:, [0, 2]].copy()
print(f'{len(xz)} census centroids')

hi = a.HighInterface()
for NM in (40, 60):
    labels = np.asarray(hi.ransac_init(xz, NM, 0.8, 1.0, 0.0, 0.985)[1])
    if labels.ndim > 1:
        labels = labels.ravel()
    n_out = int((labels == NM).sum())
    uniq = [u for u in np.unique(labels) if u != NM]
    sizes = [(int(u), int((labels == u).sum())) for u in uniq]
    sizes.sort(key=lambda t: -t[1])
    print(f'num_models={NM}: {len(uniq)} models, outlier bucket {n_out}, '
          f'top sizes {sizes[:6]}')
    # x-span per model
    wide = sum(1 for u in uniq
               if np.ptp(xz[labels == u, 0]) > 3 and (labels == u).sum() > 3)
    print(f'  wide(>3m x-span, n>3): {wide}')

# final render with NM=60
NM = 60
labels = np.asarray(hi.ransac_init(xz, NM, 0.8, 1.0, 0.0, 0.985)[1]).ravel()
fig, ax = plt.subplots(figsize=(9, 12))
uniq = sorted(u for u in np.unique(labels) if u != NM)
for i, u in enumerate(uniq):
    m = xz[labels == u]
    m = m[np.argsort(m[:, 1])]
    c = cm.tab20(i % 20)
    ax.plot(m[:, 0], m[:, 1], '-', color=c, lw=1.3, alpha=0.85, zorder=2)
    ax.scatter(m[:, 0], m[:, 1], c=[c], s=20, edgecolors='k', linewidths=0.3,
               zorder=3)
    if len(m) > 2:
        ax.annotate(str(int(u)), (m[-1, 0], m[-1, 1] + 1.0), ha='center',
                    fontsize=8, color=c, fontweight='bold')
out_pts = xz[labels == NM]
ax.scatter(out_pts[:, 0], out_pts[:, 1], marker='x', s=26, c='0.6', zorder=1,
           label=f'outlier bucket ({len(out_pts)})')
ax.legend(loc='lower left', fontsize=9)
ax.set_aspect('equal')
ax.set_xlabel('x cross-row (m)')
ax.set_ylabel('z along-row (m)')
ax.set_title(f'JULY-STYLE centroid solve — mv10 census (783), ransac_init\n'
             f'num_models={NM}, thr 0.8, dir 0.985 — outlier bucket EXCLUDED from rows')
ax.grid(alpha=0.25)
o = '/home/paperspace/code/lab_notebook/figs/topdown_apr_july_centroid_solve.png'
plt.tight_layout()
plt.savefig(o, dpi=130)
print('saved', o)
