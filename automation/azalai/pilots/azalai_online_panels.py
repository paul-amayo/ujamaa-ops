#!/usr/bin/env python3
"""azalai_online_panels.py (A100 agent, 11 Nov 2026): AZ2 online pilot, the eye check of the design ("at least twelve
targets across outcomes drawn in plan view (stop, aim point, census bags, partner) and looked at").
Draws targets of variant M chosen by a fixed rule (seed 0): from the pair of record four right, four neighbour and four
other (one per lane where possible), and from each short-gap rung one right, one neighbour, one other. Each panel is the
revisit's own lane frame around the partner: lateral offset across, driving station along (up = the driving sense).
Shown: the revisit census's inner-row bags of the lane (grey), the partner (green ring), the registry target carried into
the revisit frame by the site-wide fits (black cross), the robot's true position at M's stop (blue triangle), the aim
point d further along the row (blue star), the arrival (orange dot when it is not the partner) and the ideal stop
(grey tick). Lane frames as in azalai_online_resim.py (same author, from the design text).
  azalai_online_panels.py --records <dir> --out <png>"""
import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--records', required=True); ap.add_argument('--out', required=True); A0 = ap.parse_args()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from azalai_online_geom import load_pass, placed
RD = Path(A0.records)


def pick(tag, quota, seed=0):
    rec = [json.loads(l) for l in open(RD / f'online_targets_{tag}.jsonl')]
    ok = [r for r in rec if 'skip' not in r and r['stop_evaluable'] and 'outcome' in r['var']['M']]
    rng = random.Random(seed); out = []
    for oc, n in quota:
        pool = [r for r in ok if r['var']['M']['outcome'] == oc]; rng.shuffle(pool); used = set(); got = []
        for r in pool:                      # one per lane first
            if len(got) < n and r['lane'] not in used: got.append(r); used.add(r['lane'])
        for r in pool:                      # then any lane
            if len(got) < n and r not in got: got.append(r)
        out += [(tag, r) for r in got]
    return out


sel = pick('dec_apr1130', [('right', 4), ('neighbour', 4), ('other', 4)]) + pick('apr1130_apr1400', [('right', 1), ('neighbour', 1), ('other', 1)]) + \
      pick('apr1130_apr16_1230', [('right', 1), ('neighbour', 1), ('other', 1)])
fig, axs = plt.subplots(3, 6, figsize=(16.5, 10.5), dpi=130); axs = axs.ravel(); listing = []
for ax, (tag, r) in zip(axs, sel):
    reg, rev = ('dec', 'apr1130') if tag == 'dec_apr1130' else tag.split('_', 1)
    Rg, Rv = load_pass(reg), load_pass(rev); i = r['lane'] - 1; lr = Rv['lanes'][i]; o, axis = lr['frame']; s = lr['sense']
    nrm = np.array([0.0, -1.0, 0.0]); lat_dir = np.cross(nrm, axis); lat_dir -= (lat_dir @ axis) * axis; lat_dir /= np.linalg.norm(lat_dir)
    def xy(p): p = np.asarray(p, float); return float((p - o) @ lat_dir) * s * -1.0, float(s * ((p - o) @ axis))   # x: + = right of travel
    Ginv = np.linalg.inv(placed(rev)) @ placed(reg); t_rev = Ginv[:3, :3] @ Rg['B'][r['registry_bag']] + Ginv[:3, 3]
    v = r['var']['M']; ps = r['partner_station_rev_m']; D = json.load(open(RD / f'online_summary_{tag}.json'))['standoff_d_m']
    for b, left in zip(lr['ids'], lr['left']):
        x, y = xy(Rv['B'][b])
        if abs(y - ps) < 4.0: ax.scatter(x, y, s=22, color='#9a988f' if left == (r['side'] == 'left') else '#cfcdc4', zorder=2)
    px, py = xy(Rv['B'][r['partner']]); ax.scatter(px, py, s=150, facecolor='none', edgecolor='#008300', lw=1.8, zorder=4)
    tx, ty = xy(t_rev); ax.scatter(tx, ty, marker='x', s=50, color='#0b0b0b', lw=1.4, zorder=5)
    C = Rv['c2w']; sx = sy = None
    if not v['no_stop']:
        k, f = v['stop_k'], v['stop_frac']; pos = C[k, :3, 3] if (f >= 1.0 or k == lr['k0']) else (1 - f) * C[k - 1, :3, 3] + f * C[k, :3, 3]
        sx, sy = xy(pos); ax.scatter(sx, sy, marker='^', s=60, color='#2a78d6', zorder=6); ax.scatter(sx, sy + D, marker='*', s=110, color='#2a78d6', zorder=6)
        ax.plot([sx, sx], [sy, sy + D], color='#2a78d6', lw=1.0, ls=':', zorder=3)
        if v['arrival'] != r['partner']: axx, axy = xy(Rv['B'][v['arrival']]); ax.scatter(axx, axy, s=40, color='#eb6834', zorder=5)
    ax.plot([-0.25, 0.25], [ps - D, ps - D], color='#898781', lw=1.5, zorder=3)
    tr = np.array([xy(C[kk, :3, 3]) for kk in range(lr['k0'], lr['k1'] + 1)]); ax.plot(tr[:, 0], tr[:, 1], color='#c3c2b7', lw=0.8, zorder=1)
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(ps - 3.5, ps + 3.0); ax.set_aspect('equal'); ax.tick_params(labelsize=6.5)
    miss = 'no stop' if v['no_stop'] else f"miss {v['miss_m']:+.2f} m"
    ax.set_title(f"{tag.replace('_', ' > ')} lane {r['lane']}, {'-x' if r['side'] == 'left' else '+x'} side\nbag {r['registry_bag']} > {r['partner']}: {v['outcome']}, {miss}", fontsize=7.5, loc='left')
    listing.append(dict(tag=tag, lane=r['lane'], side=r['side'], registry_bag=r['registry_bag'], partner=r['partner'], arrival=v.get('arrival'), outcome=v['outcome'], miss_m=v.get('miss_m'),
                        partner_xy=[px, py], target_xy=[tx, ty], stop_xy=[sx, sy], aim_xy=None if sy is None else [sx, sy + D], err_at_stop=v.get('station_error_at_stop_m')))
for ax in axs[len(sel):]: ax.axis('off')
fig.suptitle('AZ2 eye check, variant M: plan view in the revisit lane frame (x: right of travel, y: driving station); green ring partner, black cross registry target, '
             'blue triangle true stop, blue star aim point, orange arrival, grey tick ideal stop', fontsize=8.5, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(A0.out, facecolor='white'); json.dump(listing, open(str(A0.out).replace('.png', '.json'), 'w'), indent=1)
print('wrote', A0.out, len(sel), 'panels')
for e in listing: print(e['tag'], e['lane'], e['side'], e['registry_bag'], e['partner'], e['arrival'], e['outcome'], e['miss_m'] if e['miss_m'] is None else round(e['miss_m'], 2), 'partner', np.round(e['partner_xy'], 2), 'target', np.round(e['target_xy'], 2), 'stop', e['stop_xy'] if e['stop_xy'][0] is None else np.round(e['stop_xy'], 2), 'err_at_stop', e['err_at_stop'] if e['err_at_stop'] is None else round(e['err_at_stop'], 2))
