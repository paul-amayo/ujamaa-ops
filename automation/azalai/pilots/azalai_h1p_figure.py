#!/usr/bin/env python3
"""azalai_h1p_figure.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3): the one figure. Per category, the instructions
correct in each arm (twin, enumerated) under the frozen gold, in four panels: the original wording as scored in v1 (seed 0), the
same wording rerun tonight (the reproduction control), paraphrase set A and paraphrase set B; beside each category its difference
and the clause of the H1 rule it enters (parity |twin - enumerated| <= 1 for the five ICRA categories above the line; extension
twin > enumerated for identity, count and change below it). Drawn from the recount's output (azalai_h1p_recount.py).
usage: python3 azalai_h1p_figure.py <recount.json> <out.png>"""
import sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rc, out = sys.argv[1:3]
X = json.load(open(rc))
CATS = [('direct scene graph concepts', 'direct'), ('unambiguous goal command', 'unambiguous'), ('spatial relation disambiguation', 'spatial'),
        ('region-level disambiguation', 'region'), ('failure', 'failure'), ('identity', 'identity'), ('count', 'count'), ('change', 'change')]
NEWC = ('identity', 'count', 'change')
PANELS = [('v1', 'Original wording, v1 seed 0 (11 Nov, 15:19 UTC)'), ('control', 'Original wording, control tonight (cold start)'),
          ('A', 'Paraphrase set A: a command again'), ('B', 'Paraphrase set B: another sentence form')]
SURF, INK, INK2, GRID, TRACK = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#ecebe7'
TWIN, ENUM = '#2a78d6', '#eb6834'           # categorical slots 1 and 2 of the reference palette (validated adjacent and all-pairs)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK})
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharey=True, facecolor=SURF)
for ax, (k, title) in zip(axes.flat, PANELS):
    T = X['frozen_gold'][k]; ax.set_facecolor(SURF)
    for y, (c, short) in enumerate(CATS):
        row = T['per_category'][c]; n, t, e = row['n'], row['twin'], row['enumerated']; d = t - e
        ax.plot([0, n], [y, y], color=TRACK, lw=7, solid_capstyle='round', zorder=1)
        if t != e:
            ax.plot([min(t, e), max(t, e)], [y, y], color=INK2, lw=1, zorder=2)
        off = 0.0 if t != e else 0.17
        ax.scatter([t], [y - off], s=58, color=TWIN, edgecolors=SURF, linewidths=2, zorder=3)
        ax.scatter([e], [y + off], s=58, color=ENUM, edgecolors=SURF, linewidths=2, zorder=3)
        ok = (d > 0) if c in NEWC else (abs(d) <= 1)
        ax.text(11.6, y, f'{d:+d}', ha='right', va='center', color=INK, fontsize=9, fontweight='normal' if ok else 'bold')
        ax.text(11.85, y, 'holds' if ok else 'fails', ha='left', va='center', color=INK2 if ok else INK, fontsize=8.5, fontweight='normal' if ok else 'bold')
    ax.axhline(4.5, color=INK2, lw=0.8, zorder=0)
    ax.set_xlim(-0.4, 13.4); ax.set_xticks(range(0, 11, 2)); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    tot = T['total']; v = T['verdict']
    fails = [x for x, ok in (('parity', v['parity_two_sided']), ('extension', v['extension'])) if not ok]
    verdict = 'H1 rule holds' if v['H1_holds'] else 'H1 rule fails (' + ', '.join(fails) + ')'
    ax.set_title(f"{title}\ntwin {tot['twin']}, enumerated {tot['enumerated']} of {tot['n']}; {verdict}", fontsize=9.5, loc='left', color=INK)
    ax.text(11.6, -0.85, 'diff', ha='right', va='center', color=INK2, fontsize=8)
    ax.text(11.85, -0.85, 'clause', ha='left', va='center', color=INK2, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('instructions correct (grey track: n in the category)')
for ax in axes[:, 0]:
    ax.set_yticks(range(len(CATS)))
    ax.set_yticklabels([f"{s} ({X['frozen_gold']['v1']['per_category'][c]['n']})" for c, s in CATS])
axes[0, 0].set_ylim(len(CATS) - 0.4, -1.2)
h1 = plt.Line2D([], [], marker='o', ls='', color=TWIN, markersize=7, label='twin registry')
h2 = plt.Line2D([], [], marker='o', ls='', color=ENUM, markersize=7, label='enumerated (ICRA 2026 baseline)')
fig.legend(handles=[h1, h2], loc='upper right', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.985))
fig.suptitle('H1 grounding, frozen gold (Gemma 4 12B, temperature 0)\nabove the line: parity, |diff| <= 1; below: extension, diff > 0',
             x=0.01, y=0.995, ha='left', fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(out, dpi=160, facecolor=SURF)
print('wrote', out)
