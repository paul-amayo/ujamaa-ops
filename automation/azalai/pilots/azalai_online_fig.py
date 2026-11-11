#!/usr/bin/env python3
"""azalai_online_fig.py (A100 agent, 11 Nov 2026): AZ2 online pilot, the figure online_example.png.
Top row, the pair of record (December registry, April 15 11:39 revisit): station error against distance into the lane for
a (dead reckoning), M (causal matching, laser odometry, primary) and b (causal matching on the graph poses, i.e. perfect
odometry), ten lanes each, with +-S/2 (half the registry's bag spacing) as the band the error must stay inside. Bottom row:
the short-gap rungs S1 and S2 (April 15 11:39 registry) for M, and the error at M's first belief per lane for every read
pair. A lane is drawn orange when its first belief is more than S/2 off (the design's datum failure), blue otherwise.
Reads only online_curves_*.json and online_summary_*.json.
  azalai_online_fig.py --records <dir> --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ap = argparse.ArgumentParser(); ap.add_argument('--records', required=True); ap.add_argument('--out', required=True); A = ap.parse_args()
RD = Path(A.records)
HELD, FAIL = '#2a78d6', '#eb6834'                      # reference palette slots 1 and 2 (validated pair)
INK, INK2, MUTED, GRID, BAND = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#efeee9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def curves(tag):
    return json.load(open(RD / f'online_curves_{tag}.json'))


def summary(tag):
    return json.load(open(RD / f'online_summary_{tag}.json'))


def panel(ax, tag, var, title, ylim=(-3.8, 2.3), note=None):
    c = curves(tag); s = summary(tag); half = c['half_spacing_m']
    ax.axhspan(-half, half, color=BAND, zorder=0, lw=0)
    for y in (-half, half): ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color='#c3c2b7', lw=0.6, zorder=1)
    n_fail = 0
    for lane in c['lanes']:
        v = lane['var'].get(var)
        if v is None: continue
        e = np.array([np.nan if x is None else x for x in v['err']], float); L = np.array(lane['L'], float); ok = np.isfinite(e)
        if not ok.any(): continue
        fail = abs(e[ok][0]) > half; n_fail += fail
        ax.plot(L[ok], e[ok], color=FAIL if fail else HELD, lw=1.3, alpha=0.95, zorder=3 if fail else 2, solid_capstyle='round')
    r = s['variants'][var]
    ax.set_title(title, pad=15)
    sub = ax.text(0.0, 1.015, f"right plant {r['right']}/{r['n']} = {r['rate']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}];  datum failures {n_fail} of {len(c['lanes'])} lanes",
                  transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color=INK2)
    sub.set_in_layout(False)
    if note: ax.annotate(note[0], xy=note[1], xytext=note[2], fontsize=7.5, color=INK2, va='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7, shrinkA=2))
    ax.set_ylim(*ylim); ax.set_xlim(0, 46); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('distance into the lane (m)')


fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.6), dpi=150)
gap_note = ('lane 6: 3.6 s gap in the plain laser odometry', (31.2, -3.15), (1.5, -3.68))
panel(axs[0, 0], 'dec_apr1130', 'a', 'a  dead reckoning from the first bag seen', note=gap_note)
panel(axs[0, 1], 'dec_apr1130', 'M', 'M  causal matching, laser odometry (primary)', note=gap_note)
panel(axs[0, 2], 'dec_apr1130', 'b', 'b  causal matching, graph poses (perfect odometry)')
panel(axs[1, 0], 'apr1130_apr1400', 'M', 'S1  M, April 15 11:39 registry -> April 15 14:00')
panel(axs[1, 1], 'apr1130_apr16_1230', 'M', 'S2  M, April 15 11:39 registry -> April 16 12:30')
for ax in axs[:, 0]: ax.set_ylabel('believed minus true station (m)')
# first-belief error per lane, every read pair
ax = axs[1, 2]
pairs = [('dec_apr1130', 'o', -0.24, 'December -> April 15 11:39 (record)'), ('dec_apr1400', 's', -0.12, 'December -> April 15 14:00'),
         ('dec_apr16_1230', 'D', 0.0, 'December -> April 16 12:30'), ('apr1130_apr1400', '^', 0.12, 'S1  April 15 11:39 -> 14:00'),
         ('apr1130_apr16_1230', 'v', 0.24, 'S2  April 15 11:39 -> April 16 12:30')]
half_all = []
for tag, mk, dx, label in pairs:
    s = summary(tag); half = s['half_spacing_m']; half_all.append(half)
    for l in s['lanes']:
        v = l['var'].get('M')
        if not v or 'err_first_m' not in v: continue
        e = v['err_first_m']; ax.scatter(l['lane'] + dx, e, marker=mk, s=26, color=FAIL if abs(e) > half else HELD, edgecolor='white', linewidth=0.6, zorder=3)
hmax = max(half_all); hmin = min(half_all)
ax.axhspan(-hmin, hmin, color=BAND, zorder=0, lw=0)
for y in (-hmin, hmin): ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color='#c3c2b7', lw=0.6, zorder=1)
for x in np.arange(1.5, 10.5, 1.0): ax.axvline(x, color=GRID, lw=0.6, zorder=0)
ax.set_xlim(0.5, 10.5); ax.set_xticks(range(1, 11)); ax.set_ylim(-2.3, 2.0); ax.set_xlabel('lane'); ax.set_title('M  error at the first belief, every read pair', pad=4)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
mk_handles = [Line2D([], [], ls='', marker=mk, color=INK2, markersize=5, label=label) for _, mk, _, label in pairs]
ax.legend(handles=mk_handles, loc='upper right', fontsize=6.8, frameon=True, framealpha=0.9, edgecolor='none', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
handles = [Line2D([], [], color=HELD, lw=2, label='first belief within S/2 of the truth'), Line2D([], [], color=FAIL, lw=2, label='first belief more than S/2 off (datum failure)'),
           Patch(facecolor=BAND, edgecolor=MUTED, ls='--', label='inside +-S/2 (half the bag spacing, 0.50 to 0.51 m)')]
fig.legend(handles=handles, loc='upper left', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.005, 0.968))
fig.suptitle('AZ2 online along-row station on logs: station error per keyframe, ten lanes per panel', x=0.01, y=0.995, ha='left', fontsize=11, fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.2)
Path(A.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(A.out, facecolor='white'); print('wrote', A.out)
