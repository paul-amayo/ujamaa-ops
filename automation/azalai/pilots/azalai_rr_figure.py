#!/usr/bin/env python3
"""azalai_rr_figure.py (11 Nov 2026, A100 session) — the one summary figure of AZ1 (row to relevancy without a keyframe), drawn
from the records alone: (a) the pair of record, rule R's right-plant rate (twin of the letter) against the along-row error e of the
believed pose, per datum, with the exact 95 % interval of the primary datum and the 0.80 line of the design; (b) the ladder at
e = 0, right-plant rate with exact 95 % intervals per rung and datum; (c) the datum table: per lane, how far the oracle's row end
lies from the primary one, in median bag spacings, for every pair of passes (a cell over 0.5 is a datum failure). nerf_new:
  azalai_rr_figure.py <out.png>"""
import sys, json, numpy as np
from pathlib import Path
from scipy.stats import binomtest
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
REC = Path('/mnt/shared/azalai_rr/records'); GEO = Path('/mnt/shared/azalai_rr/geom'); OUT = sys.argv[1]
COL = dict(primary='#2a78d6', sequence='#eb6834', oracle='#1baf7a', sitewide='#eda100'); INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NAME = dict(primary='primary (first bag)', sequence='sequence (bag pattern)', oracle='oracle row end (site-wide fit)', sitewide='site-wide fit, six DoF')
def ci(k, n): c = binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact'); return c.low, c.high
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(16, 9.2), facecolor='#fcfcfb'); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.18)
# (a) the pair of record against e
ax = fig.add_subplot(gs[0, 0]); S = json.load(open(REC / 'rr_summary_dec_apr1130.json'))['part2']; O = S['outcomes']['all']; ES = sorted(float(e) for e in O['primary']['canon'])
for dn in ('primary', 'sequence', 'oracle', 'sitewide'):
    k = np.array([O[dn]['canon'][f'{e:+.2f}']['right plant'] for e in ES]); n = O[dn]['canon']['+0.00']['n']
    ax.plot(ES, k / n, '-o', color=COL[dn], lw=2, ms=6, mec='#fcfcfb', mew=1.2, label=NAME[dn], zorder=3)
    if dn == 'primary':
        lo, hi = zip(*[ci(int(x), n) for x in k]); ax.fill_between(ES, lo, hi, color=COL[dn], alpha=0.14, lw=0, zorder=1)
ax.axhline(0.80, color=INK, lw=1, ls='--'); ax.text(0.92, 0.815, '0.80 line (lower bound must clear it)', ha='right', va='bottom', fontsize=8, color=INK)
ax.set_xticks([-0.9, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.9]); ax.set_xticklabels(['-0.90', '-0.45', '-0.30', '-0.15', '0', '+0.15', '+0.30', '+0.45', '+0.90'], fontsize=8)
ax.set_ylim(0, 1); ax.set_xlabel('along-row error e of the believed pose (m, + = driving sense)'); ax.set_ylabel('right-plant rate of rule R (twin of the letter)')
ax.grid(axis='y', color=GRID, lw=0.8); ax.legend(loc='lower center', fontsize=8, frameon=False, ncol=2)
n0 = O['primary']['canon']['+0.00']['n']; ax.set_title(f'(a) Pair of record: December twin, April 15 11:39 revisit, {n0} evaluable pairs\n(band: exact 95 % interval of the primary datum)', fontsize=10, loc='left', color=INK)
# (b) the ladder at e = 0
ax = fig.add_subplot(gs[0, 1]); rungs = [('dec_apr1130', 'Apr 15 11:39\npair of record', True), ('dec_apr1130_allmnn', 'Apr 15 11:39\nall pairs\n(sensitivity)', False), ('dec_apr1400', 'Apr 15 14:00', True), ('dec_apr16_1230', 'Apr 16 12:30', True),
                                       ('dec_apr1054', 'Apr 15 10:54\nnot read', False), ('dec_feb1003', 'Feb 10:03\nnot read', False), ('dec_feb1224', 'Feb 12:24\nnot read', False)]
rungs = [r for r in rungs if (REC / f'rr_summary_{r[0]}.json').exists()]; ns = []
for i, (s, lab, solid) in enumerate(rungs):
    O2 = json.load(open(REC / f'rr_summary_{s}.json'))['part2']['outcomes']['all']
    for j, dn in enumerate(('primary', 'sequence', 'oracle')):
        c = O2[dn]['canon']['+0.00']; k, n = c['right plant'], c['n']; lo, hi = ci(k, n); x = i + (j - 1) * 0.24
        ax.errorbar([x], [k / n], yerr=[[k / n - lo], [hi - k / n]], fmt='o', color=COL[dn], mfc=COL[dn] if solid else '#fcfcfb', ms=7, mec=COL[dn] if not solid else '#fcfcfb', mew=1.4, elinewidth=2, capsize=0, label=NAME[dn] if i == 0 else None, zorder=3)
        ax.text(x, hi + 0.015, f'{k}', ha='center', va='bottom', fontsize=7, color=INK2)
    ns.append(n)
ax.axhline(0.80, color=INK, lw=1, ls='--'); ax.set_xticks(range(len(rungs))); ax.set_xticklabels([f'{r[1]}\nn = {n}' for r, n in zip(rungs, ns)], fontsize=8); ax.set_xlim(-0.6, len(rungs) - 0.4)
ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel('right-plant rate at e = 0, exact 95 % interval'); ax.grid(axis='y', color=GRID, lw=0.8); ax.legend(loc='upper center', fontsize=7.5, frameon=False, ncol=3, handletextpad=0.2, columnspacing=1.0)
ax.set_title('(b) Ladder, December twin, e = 0 (label = right-plant count; hollow =\nnot read or sensitivity; passes never pooled; A rungs not run)', fontsize=10, loc='left', color=INK)
# (c) the datum table
ax = fig.add_subplot(gs[1, :]); pairs = [('dec_apr1130_allmnn', 'December -> Apr 15 11:39 (pair of record)'), ('dec_apr1400_allmnn', 'December -> Apr 15 14:00'), ('dec_apr16_1230_allmnn', 'December -> Apr 16 12:30'),
                                         ('apr1130_apr1400', 'S1  Apr 15 11:39 -> 14:00 (2.5 h)'), ('apr1130_apr16_1230', 'S2  Apr 15 11:39 -> Apr 16 12:30 (1 day)')]
M = []; F = []
for s, _ in pairs:
    G = json.load(open(GEO / s / f'rr_geom_{s}.json'))['summary']; sp = G['spacing_m']; L = {l['lane']: l for l in G['lanes']}
    M.append([abs(L[i]['row_end_minus_oracle_m']['primary']) / sp for i in range(1, 11)]); F.append([bool(L[i]['datum_failure']) for i in range(1, 11)])
M = np.array(M); cmap = LinearSegmentedColormap.from_list('blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']); im = ax.imshow(np.clip(M, 0, 3), cmap=cmap, vmin=0, vmax=3, aspect='auto')
for r in range(M.shape[0]):
    for c in range(10):
        ax.text(c, r, f'{M[r, c]:.2f}', ha='center', va='center', fontsize=9, color='#fcfcfb' if M[r, c] > 1.3 else INK, fontweight='bold' if F[r][c] else 'normal')
        if F[r][c]: ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec='#e34948', lw=2.2))
ax.set_xticks(range(10)); ax.set_xticklabels([f'lane {i}' for i in range(1, 11)]); ax.set_yticks(range(len(pairs))); ax.set_yticklabels([l for _, l in pairs], fontsize=8.5)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('bag spacings', fontsize=8); cb.ax.tick_params(labelsize=8)
ax.set_title('(c) Datum table: how far the primary row end (first inner-row bag met) lies from the oracle\'s, in median inner-row bag spacings; red frame = datum failure (over 0.5)', fontsize=10, loc='left', color=INK)
fig.savefig(OUT, dpi=110, facecolor=fig.get_facecolor(), bbox_inches='tight'); print('[figure]', OUT)
