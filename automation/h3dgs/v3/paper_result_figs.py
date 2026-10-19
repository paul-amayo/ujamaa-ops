#!/usr/bin/env python3
"""paper_result_figs.py (2026-10-17) — the HiGH paper's result figures from numbers/*.json: fig_iou_by_size.pdf (E1 strata), fig_arms.pdf (E2
geometry arms), fig_seeding.pdf (E2b seeding arms), fig_e1_survey.pdf (Table I per survey), fig_e3.pdf (E3 category bars), fig_fruit.pdf (counts and per-apple scores). Fonts sized for the printed column (9 pt body): 8-9 pt in a 3.5-in figure.
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

# E1 per survey: held-out tree and row IoU, argmax as bars, oracle and fixed thresholds as markers
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), sharey=True); SV2 = [sv for sv in SV if sv in e1['surveys']]
for ax, lvl, title in zip(axes, ['TREE', 'ROW'], ['trees', 'rows']):
    x = np.arange(len(SV2)); g = lambda k: [e1['surveys'][sv].get(lvl, {}).get('eval', {}).get(k, np.nan) for sv in SV2]
    ax.bar(x, g('argmax_med'), 0.62, color=COLS[0], label='argmax'); ax.plot(x, g('oracle_med'), 'v', color=COLS[2], ms=4, label='oracle'); ax.plot(x, g('fixed_cell_med'), '_', color='k', ms=9, mew=1.2, label='fixed')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[sv].replace('citrus ', 'c').replace('Klapmuts ', 'K.').replace('Lindendhof', 'Lind.') for sv in SV2], fontsize=6, rotation=60, ha='right'); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
axes[0].set_ylabel('held-out IoU (median)', fontsize=8); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6.5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.0)); fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.93)); fig.savefig(F / 'fig_e1_survey.pdf'); print('fig_e1_survey')
# E3: held-out "tree" IoU at relevancy >= 0.5 (bars) and at the oracle threshold (markers), three fields
e3 = json.load(open(N / 'e3_category.json')); rows = {(r['survey'], r['level']): r for r in e3['arms']['rows']}; tags = ['cat_v3', 'cat_flat96', 'cat_lite']; names = ['HiGH', 'flat 3×32', 'LangSplat-style']
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for i, t in enumerate(tags):
    ax.bar(np.arange(5) + (i - 1) * 0.27, [rows[(sv, 'CAT:tree')]['rel05_values'][t] for sv in CIT], 0.27, color=[COLS[0], COLS[2], COLS[3]][i], label=names[i])
    ax.plot(np.arange(5) + (i - 1) * 0.27, [rows[(sv, 'CAT:tree')]['oracle_values'][t] for sv in CIT], 'v', color='k', ms=3)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(XT); ax.set_xlabel('citrus survey', fontsize=8); ax.set_ylabel('held-out "tree" IoU', fontsize=8); ax.set_ylim(0, 1)
ax.set_ylim(0, 1.18); ax.legend(fontsize=6.5, frameon=False, loc='upper center', ncol=3, title='bars: relevancy ≥ 0.5; markers: oracle threshold', title_fontsize=6); fig.tight_layout(); fig.savefig(F / 'fig_e3.pdf'); print('fig_e3')
# fruit: FruitNeRF counts per tree and Fuji per-apple scores against the references
fr = json.load(open(N / 'fruit_registry.json')); fn = fr['fruitnerf']; fj = fr['fuji_sfm']
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]; trees = ['tree_01', 'tree_02', 'tree_03']; series = [('hand count', [fn['hand_counts'][t] for t in trees], '0.3'), ('ours', [fn['ours_record'][t] for t in trees], COLS[0]), ('FruitLangGS', [fn['literature']['FruitLangGS'][t] for t in trees], COLS[2]), ('FruitNeRF', [fn['literature']['FruitNeRF_best_per_tree'][t] for t in trees], COLS[3])]
for i, (nm, v, c) in enumerate(series): ax.bar(np.arange(3) + (i - 1.5) * 0.2, v, 0.2, color=c, label=nm)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['tree 01', 'tree 02', 'tree 03'], fontsize=7); ax.set_ylabel('apples counted', fontsize=8); ax.set_ylim(0, 360); ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2); ax.set_title('FruitNeRF trees', fontsize=8)
ax = axes[1]; r10 = fj['merged_10cm_merge']['r10']; r15 = fj['merged_15cm_merge']['r15']
for i, (nm, d) in enumerate((('10 cm', r10), ('15 cm', r15))): ax.bar(np.arange(3) + (i - 0.5) * 0.36, [d['P'], d['R'], d['F1']], 0.36, color=[COLS[0], COLS[1]][i], label=f'ours, {nm}')
f1a, f1b = fj['literature']['FruitNeRF++']['F1'], fj['literature']['Gene-Mola_2020_CEA']['F1']
ax.axhline(f1a, color=COLS[3], lw=1, ls='--'); ax.text(2.4, f1a - 0.02, f'FruitNeRF++ {f1a:.2f}', fontsize=5.5, ha='left', va='top', color=COLS[3])
ax.axhline(f1b, color='0.3', lw=1, ls=':'); ax.text(2.4, f1b + 0.02, f'in-cloud {f1b:.2f}', fontsize=5.5, ha='left', va='bottom', color='0.3')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['P', 'R', 'F1'], fontsize=7); ax.set_xlim(-0.6, 3.9); ax.set_ylim(0, 1.2); ax.set_title('Fuji-SfM per apple', fontsize=8); ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2)
for a_ in axes: a_.tick_params(labelsize=6.5)
fig.tight_layout(w_pad=0.6); fig.savefig(F / 'fig_fruit.pdf'); print('fig_fruit')
