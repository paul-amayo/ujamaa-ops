#!/usr/bin/env python3
"""azalai_rr_geom_panel.py (11 Nov 2026, A100 session) — the row ends of every lane for one (registry, revisit) pair of
azalai_rr_geom.py, for the eye check of the datum: per lane, the first metres of the inner rows in the driving sense, in the
REGISTRY's lane frame (x = distance from the registry's primary row end in the driving sense, y = lateral offset): the registry's
inner-row bags (black dots), the revisit's inner-row bags carried by the primary lane-frame transfer (red plus) and by the site-wide
fit (blue rings), the revisit's own first bag carried by the site-wide fit (blue star). A datum failure shows as the red and blue
marks one bag apart. No outcome is computed here. nerf_new python3.10:
  azalai_rr_geom_panel.py <rr_geom<suffix>.json> <out.png> [--span 9]"""
import argparse, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('geom'); ap.add_argument('out'); ap.add_argument('--span', type=float, default=9.0); a = ap.parse_args()
J = json.load(open(a.geom)); S = J['summary']; SK = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/sankofa/jirs_20261023')
cd = lambda n: SK / ('camera_depth_feb_fixedreader' if n.startswith('feb') else 'lidar_fixedreader')
def G_of(n): return np.eye(4) if n == 'dec' else np.load(cd(n) / f'placed_{n}.npz')['G_global'].astype(float)
Bt = np.load(cd(S['registry']) / f"census_{S['registry']}.npz")['bags_xyz'].astype(float); Br = np.load(cd(S['revisit']) / f"census_{S['revisit']}.npz")['bags_xyz'].astype(float)
G = np.linalg.inv(G_of(S['registry'])) @ G_of(S['revisit']); tf = lambda T, p: (T[:3, :3] @ np.asarray(p, float).T).T + T[:3, 3]
fig, axs = plt.subplots(5, 2, figsize=(16, 11.5), squeeze=False)
for i, l in enumerate(J['lanes_full']):
    ax = axs[i % 5, i // 5]; ln = l['lane']
    if 'skip' in l: ax.set_title(f'lane {ln}: no lane frame'); ax.axis('off'); continue
    Ft = np.array(l['F_twin']); Fp = np.array(l['F_rev']['primary']); sn = l['sense']; T = Ft @ np.linalg.inv(Fp)
    def coords(P): q = (np.asarray(P, float) - Ft[:3, 3]); return sn * (q @ Ft[:3, 2]), q @ Ft[:3, 0]
    it = np.array(J['twin_inner'][str(ln)]['bags']); ir = np.array(J['rev_inner'][str(ln)]['bags']); x, y = coords(Bt[it]); ax.scatter(x, y, s=22, c='k', zorder=3, label=f"{S['registry']} inner-row bags")
    x, y = coords(tf(T, Br[ir])); ax.scatter(x, y, s=60, c='tab:red', marker='+', linewidths=1.4, zorder=4, label=f"{S['revisit']} bags, primary lane frames")
    x, y = coords(tf(G, Br[ir])); ax.scatter(x, y, s=70, facecolors='none', edgecolors='tab:blue', linewidths=1.1, zorder=2, label=f"{S['revisit']} bags, site-wide fit")
    fb = l['rev_inner']['first_bag']; x, y = coords(tf(G, Br[[fb]])); ax.scatter(x, y, s=150, c='tab:blue', marker='*', zorder=5, label=f"{S['revisit']} first bag (its row end), site-wide fit")
    ax.axvline(0, c='k', lw=0.8, ls='--'); off = l['row_end_minus_oracle_m']['primary']; fail = l['datum_failure']
    ax.set_xlim(-2.5, a.span); ax.set_ylim(-1.9, 1.9); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    ax.set_title(f"lane {ln} ({'+z' if sn > 0 else '-z'}): oracle row end {off:+.2f} m from the primary one ({abs(off) / S['spacing_m']:.2f} bag spacings)" + ('  DATUM FAILURE' if fail else ''), fontsize=9, color='tab:red' if fail else 'k')
    if i % 5 == 4: ax.set_xlabel(f"distance from the {S['registry']} primary row end in the driving sense (m)", fontsize=9)
    ax.set_ylabel('lateral (m)', fontsize=8)
axs[0, 0].legend(loc='upper right', fontsize=7.5, ncol=2, framealpha=0.9)
fig.suptitle(f"Row ends, {S['registry']} registry against the {S['revisit']} revisit, in the registry's lane frames (spacing {S['spacing_m']:.3f} m; datum failure = over half a spacing)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(a.out, dpi=85); print('[geom-panel]', a.out)
