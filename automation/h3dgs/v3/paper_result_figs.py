#!/usr/bin/env python3
"""paper_result_figs.py (2026-10-17) — the HiGH paper's result figures from numbers/*.json: fig_iou_by_size.pdf (E1 strata), fig_arms.pdf (E2
geometry arms), fig_seeding.pdf (E2b seeding arms). Fonts sized for the printed column (9 pt body): 8-9 pt in a 3.5-in figure.
system python3:  paper_result_figs.py <paper dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = Path(sys.argv[1]); N = P / 'numbers'; F = P / 'figures'
e1 = json.load(open(N / 'e1_v3.json')); e2 = json.load(open(N / 'e2_arms.json')); e2b = json.load(open(N / 'e2b_seed.json'))
SV = ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal', '04_13D_Jackal', '05_13D_Jackal', 'apr_2026_zed', 'dec_2025_ten_rows', 'nov_2025_a300', 'IMG_7993_s0', 'IMG_7990_s1']
SHORT = {'01_13B_Jackal': 'citrus 01', '02_13B_Jackal': 'citrus 02', '03_13B_Jackal': 'citrus 03', '04_13D_Jackal': 'citrus 04', '05_13D_Jackal': 'citrus 05', 'apr_2026_zed': 'Klapmuts Apr',
         'dec_2025_ten_rows': 'Klapmuts Dec', 'nov_2025_a300': 'Lindendhof', 'IMG_7993_s0': 'cabbage', 'IMG_7990_s1': 'pepper'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COLS = ['#1b6ca8', '#8fb8d8', '#d98c3f', '#6a994e']; CIT = SV[:5]; XT = ['01', '02', '03', '04', '05']
# E1: held-out tree IoU by object size (the strata are reported on every head, the protocol floor is drawn)
fig, ax = plt.subplots(figsize=(3.5, 2.6)); strata = ['<0.2%', '0.2-1%', '1-3%', '>=3%']; x = np.arange(4); mk = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']
for i, sv in enumerate(SV):
    st = e1['surveys'].get(sv, {}).get('TREE', {}).get('eval', {}).get('strata', {}); y = [st.get(k, {}).get('argmax_med', np.nan) if st.get(k, {}).get('n', 0) >= 5 else np.nan for k in strata]
    if np.all(np.isnan(y)): continue
    ax.plot(x, y, marker=mk[i], ms=4.5, lw=1.3, label=SHORT[sv])
ax.axvspan(-0.5, 0.5, color='0.92', zorder=0); ax.text(0, 0.45, 'not counted', fontsize=7, ha='center', va='center', color='0.35', rotation=90)
ax.set_xticks(x); ax.set_xticklabels(['< 0.2 %', '0.2–1 %', '1–3 %', '≥ 3 %']); ax.set_xlabel('object size, fraction of the frame'); ax.set_ylabel('held-out tree IoU (median)'); ax.set_ylim(0, 1); ax.set_xlim(-0.5, 3.3)
ax.legend(fontsize=7, ncol=2, frameon=False, loc='upper left', bbox_to_anchor=(0.12, 1.0)); fig.tight_layout(); fig.savefig(F / 'fig_iou_by_size.pdf'); print('fig_iou_by_size')
# E2: the geometry arms
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True); arms = ['v3', 'hyper_own', 'flat96', 'euclid']; names = ['HiGH 32-d, containment', 'own point only', 'flat 3×32-d', 'Euclidean 32-d']
rows = {(r['survey'], r['level']): r for r in e2['arms']['rows']}
for ax, lvl, title in zip(axes, ['TREE', 'ROW', 'nesting'], ['trees, held-out IoU', 'rows asked by their address', 'nesting consistency']):
    for i, a_ in enumerate(arms): ax.bar(np.arange(5) + (i - 1.5) * 0.2, [rows[(sv, lvl)]['values'].get(a_, np.nan) for sv in CIT], 0.2, color=COLS[i], label=names[i])
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(XT); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1); ax.set_xlabel('citrus survey', fontsize=8)
axes[0].set_ylabel('median'); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(F / 'fig_arms.pdf'); print('fig_arms')
# E2b: the seeding arms
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.7), sharey=True); arms = ['v3', 'zero3000', 'refine3000', 'projvote']; names = ['census seed', 'learned from zero', 'seed + refine', 'projection vote']
rows = {(r['survey'], r['level']): r for r in e2b['arms']['rows']}
for ax, lvl, title in zip(axes, ['TREE', 'ROW'], ['trees', 'rows by address']):
    for i, a_ in enumerate(arms): ax.bar(np.arange(5) + (i - 1.5) * 0.2, [rows[(sv, lvl)]['values'].get(a_, np.nan) for sv in CIT], 0.2, color=COLS[i], label=names[i])
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(XT, fontsize=8); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1); ax.set_xlabel('citrus survey', fontsize=8)
axes[0].set_ylabel('held-out IoU (median)', fontsize=8); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig(F / 'fig_seeding.pdf'); print('fig_seeding')
