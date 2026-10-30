#!/usr/bin/env python3
"""azalai_registry.py (29 Oct 2026) — the registry the H1 pilot grounds against, built from data in place: Klapmuts December
census (bag id = census index), rows by across-row (x) clustering, order along the row (z), December plant-top height, the
December-to-April growth call (majority of the three full April passes against the 0.082 m same-day threshold, as in the JIRS
paper), the JIRS vetting status (found again / association miss / removed / unclear / december false positive; added bags as
extra entries), and the HiGH fruit registry (FruitNeRF trees 1-3, Fuji-SfM trees) for identity / count. Writes
Azalai_RSS/numbers/registry_v0.json. Run on the Mac from the publications tree."""
import json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
PUB = Path.home() / 'Documents/UCT/SIL/Publications'; L = PUB / 'Sankofa_JIRS/numbers/sankofa/jirs_20261023/lidar_fixedreader'; H = PUB / 'High/Hyperbolic-Hierarchies-for-Precision-Agriculture/paper-template-latex/numbers'
z = np.load(L / 'census_dec.npz'); B = z['bags_xyz'].astype(float)[:, [0, 2]]; ok = np.flatnonzero(np.isfinite(B).all(1))   # raw frame, as the vetting's xyz_dec
hd = np.load(L / 'heights_dec.npz')['height'].astype(float)
# rows: the nursery has eleven hedges (the hand annotation covers eight of them); bags stand on both faces of a hedge, so a
# 1-D k-means on x with k = 11, seeded at equal quantiles, gives the hedges; a 'row' in the registry is a hedge
from scipy.cluster.vq import kmeans2
xs = B[ok, 0]; seeds = np.quantile(xs, (np.arange(11) + 0.5) / 11)[:, None]
cent, labl = kmeans2(xs[:, None], seeds, minit='matrix', iter=50); cent = cent[:, 0]
rank = {int(c): k for k, c in enumerate(np.argsort(cent))}; row_of = {int(ok[a]): rank[int(labl[a])] + 1 for a in range(len(ok))}
rows = sorted(set(row_of.values()))
order = {}
for r in set(row_of.values()):
    ii = sorted([i for i in row_of if row_of[i] == r], key=lambda i: B[i, 1])
    for n, i in enumerate(ii, 1): order[i] = n
# growth call: majority of the three full passes
thr = json.load(open(L / 'jirs_summary.json'))['heights']['threshold_m']; calls = {int(i): [] for i in ok}; dh_all = {int(i): [] for i in ok}
for p in ('apr1130', 'apr1400', 'apr16_1230'):
    S = np.load(L / f'placed_{p}.npz')['global'].astype(float)[:, [0, 2]]; oka = np.flatnonzero(np.isfinite(S).all(1)); ha = np.load(L / f'heights_{p}.npz')['height'].astype(float)
    dA, jA = cKDTree(S[oka]).query(B[ok]); dB, jB = cKDTree(B[ok]).query(S[oka])
    for a in range(len(ok)):
        if dA[a] <= 0.5 and jB[jA[a]] == a:
            i = int(ok[a]); j = int(oka[jA[a]]); d = ha[j] - hd[i]
            if np.isfinite(d): dh_all[i].append(float(d)); calls[i].append('grown' if d > thr else ('shrunk' if d < -thr else 'stable'))
def majority(c):
    if len(c) < 2: return None
    from collections import Counter; k, n = Counter(c).most_common(1)[0]; return k if n * 2 > len(c) else 'split'
# vetting status
v = json.load(open(L / 'vet_unmatched.json')); status = {int(i): 'found again' for i in ok}; added = []
t = cKDTree(B[ok])
for c in v['candidates']:
    p = np.array(c['xyz_dec'])[[0, 2]]
    if c['kind'] == 'december_only':
        d, a = t.query(p); i = int(ok[a])
        if d < 0.5: status[i] = c['verdict']
    elif c['verdict'] == 'added': added.append(dict(xz=[round(float(p[0]), 2), round(float(p[1]), 2)], status='added'))
bags = []
for i in ok:
    i = int(i); bags.append(dict(id=i, row=row_of[i], n_in_row=order[i], x=round(float(B[i, 0]), 2), z=round(float(B[i, 1]), 2), height_dec_m=(round(float(hd[i]), 3) if np.isfinite(hd[i]) else None),
                                 dh_apr_m=(round(float(np.median(dh_all[i])), 3) if dh_all[i] else None), growth=majority(calls[i]), status=status[i]))
for k, a in enumerate(added, 1):
    d, j = t.query(a['xz']); a.update(id=f'added_{k}', row=row_of[int(ok[j])], n_in_row=None, x=a['xz'][0], z=a['xz'][1], height_dec_m=None, dh_apr_m=None, growth=None); bags.append(a)
fr = json.load(open(H / 'fruit_registry.json'))
citrus = dict(fruitnerf=dict(hand=fr['fruitnerf']['hand_counts'], ours=fr['fruitnerf']['ours_record']), fuji=fr.get('fuji_sfm'))
R = dict(site='Klapmuts blueberry nursery, December 2025 reference, April 2026 revisit', rows=len(rows), row_rule='rows (hedges) numbered 1..11 by increasing x (across-row), k-means on x; n_in_row by increasing z (along the row); bag id = December census index',
         growth_rule=f'majority of the three full April passes (11:39, 14:10, Thu 12:30) against the same-day threshold {thr} m', status_rule='JIRS vetting of unpaired candidates (support >= 3 vote, masks of every pass)',
         bags=bags, fruit=citrus)
out = PUB / 'Azalai_RSS/numbers/registry_v0.json'; json.dump(R, open(out, 'w'), indent=1)
from collections import Counter
print('bags', len(bags), 'rows', len(rows), 'per row', Counter(b['row'] for b in bags), '\nstatus', Counter(b['status'] for b in bags), '\ngrowth', Counter(b['growth'] for b in bags))
