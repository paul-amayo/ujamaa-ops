#!/usr/bin/env python3
"""paper_result_figs.py (2026-10-17) — the HiGH paper's result figures from numbers/*.json: fig_iou_by_size.pdf (E1 strata), fig_arms.pdf (E2
geometry arms), fig_seeding.pdf (E2b seeding arms), fig_e1_survey.pdf (Table I per survey), fig_e3.pdf (E3 category bars), fig_fruit.pdf (counts and per-apple scores), fig_cost.pdf (storage per cell, serving fps and query latency from numbers/e5_cost.json). Fonts sized for the printed column (9 pt body): 8-9 pt in a 3.5-in figure.
system python3:  paper_result_figs.py <paper dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = Path(sys.argv[1]); N = P / 'numbers'; F = P / 'figures'
# v39 (2026-11-11): the void-free encoder-path readout (tag tangent_vf) and its arms; pre-v39 files kept as e1_v3 / e2_arms / e2b_seed / e3_category
e1 = json.load(open(N / 'e1_tangent.json')); e2 = json.load(open(N / 'e2_arms_vf.json')); e2b = json.load(open(N / 'e2b_seed_vf.json'))
T_HIGH = 'tangent_vf'
SV = ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal', '04_13D_Jackal', '05_13D_Jackal', 'apr_2026_zed', 'dec_2025_ten_rows', 'nov_2025_a300']   # the phone walks are not part of the letter (draft 36)
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
# E2: the geometry arms (column width, three stacked panels, 8 pt text at print size; draft 36)
fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.9), sharex=True); arms = [T_HIGH, 'hyper_own_vf', 'flat96_vf', 'euclid_vf']; names = ['HiGH, containment', 'own point only', 'flat 3×32-d', 'Euclidean 32-d']
rows = {(r['survey'], r['level']): r for r in e2['arms']['rows']}
for ax, lvl, title in zip(axes, ['TREE', 'ROW', 'nesting'], ['trees', 'rows by address', 'nesting consistency']):
    for i, a_ in enumerate(arms): ax.bar(np.arange(5) + (i - 1.5) * 0.2, [rows[(sv, lvl)]['values'].get(a_, np.nan) for sv in CIT], 0.2, color=COLS[i], label=names[i])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=8); ax.text(0.01, 1.0, title, fontsize=9, transform=ax.transAxes, va='bottom', ha='left')
axes[-1].set_xticks(np.arange(5)); axes[-1].set_xticklabels(XT); axes[-1].set_xlabel('citrus survey', fontsize=9); axes[1].set_ylabel('held-out median', fontsize=9)
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
fig.tight_layout(h_pad=0.9, rect=(0, 0, 1, 0.9)); fig.savefig(F / 'fig_arms.pdf'); print('fig_arms')
# E2b: the seeding arms
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.7), sharey=True); arms = [T_HIGH, 'zero3000_tvf', 'refine3000_tvf', 'projvote_tvf']; names = ['census seed', 'learned from zero', 'seed + refine', 'projection vote']
rows = {(r['survey'], r['level']): r for r in e2b['arms']['rows']}; SV7 = CIT + ['apr_2026_zed', 'nov_2025_a300']; XT7 = ['01', '02', '03', '04', '05', 'K.Apr', 'Lind.']
for ax, lvl, title in zip(axes, ['TREE', 'ROW'], ['trees', 'rows by address']):
    for i, a_ in enumerate(arms): ax.bar(np.arange(len(SV7)) + (i - 1.5) * 0.2, [rows[(sv, lvl)]['values'].get(a_, np.nan) for sv in SV7], 0.2, color=COLS[i], label=names[i])
    ax.set_xticks(np.arange(len(SV7))); ax.set_xticklabels(XT7, fontsize=6.5, rotation=45, ha='right'); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1); ax.set_xlabel('survey', fontsize=8)
axes[0].set_ylabel('held-out IoU (median)', fontsize=8); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig(F / 'fig_seeding.pdf'); print('fig_seeding')

# E1 per survey: held-out tree and row IoU, argmax as bars, oracle and fixed thresholds as markers (column width, two stacked panels, 8 pt; draft 36)
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True); SV2 = [sv for sv in SV if sv in e1['surveys']]
for ax, lvl, title in zip(axes, ['TREE', 'ROW'], ['trees', 'rows']):
    x = np.arange(len(SV2)); g = lambda k: [e1['surveys'][sv].get(lvl, {}).get('eval', {}).get(k, np.nan) for sv in SV2]
    ax.bar(x, g('argmax_med'), 0.62, color=COLS[0], label='argmax'); ax.plot(x, g('oracle_med'), 'v', color=COLS[2], ms=5, label='oracle'); ax.plot(x, g('fixed_cell_med'), '_', color='k', ms=10, mew=1.4, label='fixed')
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=8); ax.text(0.01, 1.0, title, fontsize=9, transform=ax.transAxes, va='bottom', ha='left')
axes[-1].set_xticks(x); axes[-1].set_xticklabels([SHORT[sv] for sv in SV2], fontsize=8, rotation=40, ha='right', rotation_mode='anchor')
fig.supylabel('held-out IoU (median)', fontsize=9, x=0.02); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(h_pad=0.9, rect=(0.02, 0, 1, 0.93)); fig.savefig(F / 'fig_e1_survey.pdf'); print('fig_e1_survey')
# E3: held-out "tree" IoU at relevancy >= 0.5 (bars) and at the oracle threshold (markers), three fields
e3 = json.load(open(N / 'e3_category_vf.json')); rows = {(r['survey'], r['level']): r for r in e3['arms']['rows']}; tags = ['cat_walk_vf', 'cat_lite']; names = ['HiGH', 'lifted CLIP']
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for i, t in enumerate(tags):
    ax.bar(np.arange(5) + (i - 1) * 0.27, [rows[(sv, 'CAT:tree')]['rel05_values'][t] for sv in CIT], 0.27, color=[COLS[0], COLS[2], COLS[3]][i], label=names[i])
    ax.plot(np.arange(5) + (i - 1) * 0.27, [rows[(sv, 'CAT:tree')]['oracle_values'][t] for sv in CIT], 'v', color='k', ms=3)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(XT); ax.set_xlabel('citrus survey', fontsize=8); ax.set_ylabel('held-out "tree" IoU', fontsize=8); ax.set_ylim(0, 1)
ax.set_ylim(0, 1.18); ax.legend(fontsize=6.5, frameon=False, loc='upper center', ncol=3, title='bars: relevancy ≥ 0.5; markers: oracle threshold', title_fontsize=6); fig.tight_layout(); fig.savefig(F / 'fig_e3.pdf'); print('fig_e3')
# fruit: FruitNeRF counts per tree and Fuji per-apple scores against the references
fr = json.load(open(N / 'fruit_registry.json')); fn = fr['fruitnerf']; fj = fr['fuji_sfm']
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.7), gridspec_kw={'width_ratios': [1.15, 1]})   # 8 pt text at print size (draft 36)
ax = axes[0]; trees = ['tree_01', 'tree_02', 'tree_03']; series = [('hand', [fn['hand_counts'][t] for t in trees], '0.3'), ('ours', [fn['ours_record'][t] for t in trees], COLS[0]), ('FruitLangGS', [fn['literature']['FruitLangGS'][t] for t in trees], COLS[2]), ('FruitNeRF', [fn['literature']['FruitNeRF_best_per_tree'][t] for t in trees], COLS[3])]
for i, (nm, v, c) in enumerate(series): ax.bar(np.arange(3) + (i - 1.5) * 0.2, v, 0.2, color=c, label=nm)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['01', '02', '03']); ax.set_xlabel('FruitNeRF tree', fontsize=9); ax.set_ylabel('apples counted', fontsize=9); ax.set_ylim(0, 380)
ax.legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(-0.25, 1.02), ncol=2, columnspacing=0.8, handlelength=1.0, borderaxespad=0)
ax = axes[1]; r10 = fj['merged_10cm_merge']['r10']; r15 = fj['merged_15cm_merge']['r15']
for i, (nm, d) in enumerate((('ours, 10 cm', r10), ('ours, 15 cm', r15))): ax.bar(np.arange(3) + (i - 0.5) * 0.36, [d['P'], d['R'], d['F1']], 0.36, color=[COLS[0], COLS[1]][i], label=nm)
f1a, f1b = fj['literature']['FruitNeRF++']['F1'], fj['literature']['Gene-Mola_2020_CEA']['F1']
ax.axhline(f1a, color=COLS[3], lw=1.2, ls='--', label=f'FruitNeRF++ F1'); ax.axhline(f1b, color='0.3', lw=1.2, ls=':', label=f'in-cloud F1')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['P', 'R', 'F1']); ax.set_xlabel('Fuji-SfM, per apple', fontsize=9); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
ax.legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(-0.05, 1.02), ncol=1, handlelength=1.4, borderaxespad=0, labelspacing=0.2)
for a_ in axes: a_.tick_params(labelsize=8)
fig.tight_layout(w_pad=0.8); fig.savefig(F / 'fig_fruit.pdf'); print('fig_fruit')

# cost: storage per cell of the identity designs, serving rate and query latency (numbers/e5_cost.json)
c = json.load(open(N / 'e5_cost.json')); st = c['storage_mb']
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8), gridspec_kw={'width_ratios': [1.3, 1]})
ax = axes[0]; names = ['HiGH\n32-d', 'flat\n3×10', 'flat\n3×32', 'lifted\nCLIP']; vals = [st['features_B'], st['flat_equal_storage'], st['flat_3x32'], st['langsplat_lite']]
bars = ax.bar(np.arange(4), vals, 0.62, color=[COLS[0], COLS[1], COLS[2], COLS[3]]); ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=5.5); ax.set_ylabel('MB per cell', fontsize=7); ax.set_title(f"identity features, {c['gaussians'] / 1e6:.2f} M Gaussians", fontsize=7)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 4, f'{v:.0f}', ha='center', fontsize=6)
ax.set_ylim(0, max(vals) * 1.18); ax.tick_params(labelsize=6)
ax = axes[1]; lat = [c['render_ms'], c['round_trip_ms'], c['query_ms_row'], c['query_ms_first_tree']]; lnames = ['render', 'round\ntrip', 'row\nquery', '1st tree\nquery']
bars = ax.bar(np.arange(4), lat, 0.62, color=COLS[0]); ax.set_xticks(np.arange(4)); ax.set_xticklabels(lnames, fontsize=5.5); ax.set_ylabel('ms', fontsize=7); ax.set_title(f"serving at {c['fps']} fps", fontsize=7)
for b, v in zip(bars, lat): ax.text(b.get_x() + b.get_width() / 2, v + 4, f'{v:.0f}', ha='center', fontsize=6)
ax.set_ylim(0, max(lat) * 1.18); ax.tick_params(labelsize=6)
for a_ in axes: a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
fig.tight_layout(w_pad=0.5); fig.savefig(F / 'fig_cost.pdf'); print('fig_cost')
