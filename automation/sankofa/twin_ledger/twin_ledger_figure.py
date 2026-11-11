#!/usr/bin/env python3
"""twin_ledger_figure.py (11 Nov 2026, A100 session) - JIRS test JS-1, the one figure: twin_vs_lidar.png. (a, b) the twin-read
plant height (method b) against the lidar's height_cam for every named bag of each season, with the twin's own lane constant
(primary, filled) and with the lidar's lane constant (the design's secondary line, hollow), the diagonal and the paper's
same-day threshold of 0.082 m as a band; (c) the growth call read from the twins against the paper's call (majority of three
full April passes) on the twin pairs; (d) the lidar's own 10:54 pass against the paper's call on the same December bags.
Everything is read from the records and summary.json; the palette and fonts are the paper's (figures/make_figures.py).
  python3 twin_ledger_figure.py <records dir> <results dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REC, RES = Path(sys.argv[1]), Path(sys.argv[2]); SUM = json.load(open(RES / 'summary.json')); TH = 0.082
INK, BLUE, GREEN, OCHRE, RED, GREY = '#1E2A22', '#2F5D8A', '#1E8C5A', '#C9A227', '#B3261E', '#9AA59C'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'font.family': 'DejaVu Sans'})
plants = {s: [json.loads(l) for l in open(REC / f'plants_{s}.jsonl')] for s in ('dec', 'apr')}
fig, ax = plt.subplots(1, 4, figsize=(17.5, 5.4), gridspec_kw={'width_ratios': [1, 1, 0.8, 0.8]}); fig.subplots_adjust(0.045, 0.27, 0.99, 0.86, 0.36)
for i, (s, name) in enumerate((('dec', 'December 2025'), ('apr', 'April 15 2026, 10:54'))):
    a = ax[i]; rows = [(e['lidar_height_cam'], e['b']['height'], e['b']['height_lidarlane']) for e in plants[s]
                       if e['in_vocab'] and e.get('named_bag') is not None and e.get('lidar_height_cam') is not None and e['b'].get('height') is not None]
    Lh, Tw, Tl = (np.array(c, float) for c in zip(*rows)); T3 = SUM['T_id3'][s]; lane = SUM['lane_constant'][s]
    lo = min(Lh.min(), Tw.min(), Tl.min()) - 0.05; hi = max(Lh.max(), Tw.max(), Tl.max()) + 0.05; lim = np.array([lo, hi])
    a.fill_between(lim, lim - TH, lim + TH, color=GREEN, alpha=0.13, lw=0, label=f'±{TH} m, the paper\'s same-day threshold')
    a.plot(lim, lim, color=INK, lw=0.7)
    a.scatter(Lh, Tl, s=9, facecolors='none', edgecolors=GREY, linewidths=0.6, label=f"twin, lidar lane constant {lane['lidar']:.3f} m: p95 |d| {T3['b_height_lidarlane']['abs_p95']:.3f} m")
    a.scatter(Lh, Tw, s=7, color=BLUE, alpha=0.65, linewidths=0, label=f"twin, twin lane constant {lane['twin']:.3f} m: p95 |d| {T3['b_height']['abs_p95']:.3f} m")
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal'); a.set_xlabel('lidar plant-top height above the lane (height_cam), m'); a.set_ylabel('twin-read plant-top height, m')
    note = '' if s == 'dec' else '; the April addresses cover the grow bags only (panels)'
    a.set_title(f"({'ab'[i]}) {name}: {len(Lh)} named bags{note}\ntwin minus lidar, median {T3['b_height']['bias_median']:+.3f} m (twin lane), {T3['b_height_lidarlane']['bias_median']:+.3f} m (lidar lane)", fontsize=8.5)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7.5, frameon=False); a.grid(alpha=0.2)
lab = ['grown', 'stable', 'shrunk']
for j, (key, title) in enumerate((('twin_vs_L2', 'twin pairs, twin call'), ('L1_vs_L2', 'same bags, lidar 10:54 call'))):
    a = ax[2 + j]; blk = SUM['T_id4']['all_twin_pairs_with_paper_call'][key]; M = np.array(blk['matrix'])
    a.imshow(M, cmap='Blues', vmin=0, vmax=max(M.max(), 1))
    for r in range(3):
        for c in range(3): a.text(c, r, str(M[r, c]), ha='center', va='center', color=('white' if M[r, c] > 0.55 * M.max() else INK), fontsize=10)
    a.set_xticks(range(3)); a.set_xticklabels(lab); a.set_yticks(range(3)); a.set_yticklabels(lab); a.set_xlabel("the paper's call (majority of three full April passes)")
    a.set_ylabel('twin call (10:54 twin against December twin)' if j == 0 else 'lidar call (10:54 pass against December)')
    a.set_title(f"({'cd'[j]}) growth at ±{TH} m: {title}\n{blk['agree']} of {blk['n']} agree with the paper's call\n({100 * (blk['agreement'] or 0):.1f} %)", fontsize=8.5)
fig.suptitle('JS-1: the per-plant ledger read through the identity field of the Klapmuts twins, against the lidar ledger', fontsize=10.5)
fig.savefig(RES / 'twin_vs_lidar.png', dpi=200); print('[figure] wrote', RES / 'twin_vs_lidar.png')
