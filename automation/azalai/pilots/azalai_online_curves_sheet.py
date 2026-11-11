#!/usr/bin/env python3
"""azalai_online_curves_sheet.py (A100 agent, 11 Nov 2026): AZ2 eye check of the design ("the station-error curves of every
lane looked at"): every lane of a, M and b for the five read pairs, one row per pair, lanes labelled at their last keyframe.
Supplementary to online_example.png; reads online_curves_*.json only.
  azalai_online_curves_sheet.py --records <dir> --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--records', required=True); ap.add_argument('--out', required=True); A = ap.parse_args()
PAIRS = [('dec_apr1130', 'December > April 15 11:39'), ('dec_apr1400', 'December > April 15 14:00'), ('dec_apr16_1230', 'December > April 16 12:30'),
         ('apr1130_apr1400', 'S1 April 15 11:39 > 14:00'), ('apr1130_apr16_1230', 'S2 April 15 11:39 > April 16 12:30')]
fig, axs = plt.subplots(5, 3, figsize=(15, 17), dpi=110)
for row, (tag, name) in enumerate(PAIRS):
    c = json.load(open(Path(A.records) / f'online_curves_{tag}.json')); h = c['half_spacing_m']
    for col, v in enumerate(('a', 'M', 'b')):
        ax = axs[row, col]; ax.axhspan(-h, h, color='#efeee9', lw=0); [ax.axhline(y, color='#898781', lw=0.7, ls='--') for y in (-h, h)]
        for lane in c['lanes']:
            e = np.array([np.nan if x is None else x for x in lane['var'][v]['err']], float); L = np.array(lane['L']); ok = np.isfinite(e)
            if not ok.any(): continue
            col_ = '#eb6834' if abs(e[ok][0]) > h else '#2a78d6'; ax.plot(L[ok], e[ok], color=col_, lw=1.1)
            ax.text(L[ok][-1] + 0.3, e[ok][-1], str(lane['lane']), fontsize=7, color=col_, va='center')
        ax.set_ylim(-4.0, 2.5); ax.set_xlim(0, 49); ax.grid(axis='y', color='#e1e0d9', lw=0.5); ax.set_title(f'{name}: {v}', fontsize=9, loc='left')
        if col == 0: ax.set_ylabel('station error (m)')
        if row == 4: ax.set_xlabel('distance into the lane (m)')
fig.tight_layout(); fig.savefig(A.out, facecolor='white'); print('wrote', A.out)
