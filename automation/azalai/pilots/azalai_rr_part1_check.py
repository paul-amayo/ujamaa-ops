#!/usr/bin/env python3
"""azalai_rr_part1_check.py (11 Nov 2026, A100 session) — independent re-derivation of Part 1's "lands on the target" for a seeded
sample of targets and every pose type, by separately written arithmetic: the target's pixels in the stored label map, unprojected
through the stored expected depth with the job's camera (pixel centres, intrinsics at the render scale), the per-axis median of the
points, the nearest hierarchy anchor in the two axes the anchors spread most along; compared with the scorer's record (lit, area,
nearest anchor, landing error). Exit status 1 on any disagreement. nerf_new python3.10:
  azalai_rr_part1_check.py <prep.json> <renders dir> <rr_targets.jsonl> [--n 60]"""
import argparse, json, random, sys, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('renders'); ap.add_argument('targets'); ap.add_argument('--n', type=int, default=60); a = ap.parse_args()
P = json.load(open(a.prep)); RD = Path(a.renders); ids = sorted(int(k) for k in P['anchors']); A = np.array([P['anchors'][str(k)] for k in ids])
spread = A.max(0) - A.min(0); g = list(np.argsort(spread)[1:])                                    # drop the axis the anchors span least (up)
rec = {(r['cell'], r['twin_id']): r for r in map(json.loads, open(a.targets)) if r['part'] == 1}; pj = {(p['cell'], p['twin_id']): p for p in P['part1']}
random.seed(1); keys = random.sample(sorted(pj), min(a.n, len(pj))); bad = 0; n = 0
for key in keys:
    p = pj[key]; r = rec[key]; K = np.array(P['cells'][p['cell']]['K']) / 2.0
    for tag, q in p['poses'].items():
        z = np.load(RD / f"{q['job']}.npz"); c2w = np.array(q['c2w'])
        for tw in ('canon', 'bag'):
            if tw not in z.files: continue
            m = z[tw] == p['twin_id']; got = r['poses'][tag][tw]; n += 1
            if not m.any():
                if got['lit']: bad += 1; print('lit differs', key, tag, tw)
                continue
            vv, uu = np.where(m); d = z['depth'][vv, uu].astype(np.float64)
            X = np.stack([(uu + 0.5 - K[2]) * d / K[0], (vv + 0.5 - K[3]) * d / K[1], d], 1) @ c2w[:3, :3].T + c2w[:3, 3]
            med = np.median(X, axis=0); near = ids[int(np.argmin(((A[:, g] - med[g]) ** 2).sum(1)))]
            err = float(np.hypot(*(A[ids.index(p['twin_id'])][g] - med[g])))
            ok = got['lit'] and got['area_px'] == int(m.sum()) and got['nearest_anchor'] == near and abs(got['err_m'] - err) < 1e-6
            if not ok: bad += 1; print('DISAGREE', key, tag, tw, 'mine', int(m.sum()), near, round(err, 4), 'stored', got.get('area_px'), got.get('nearest_anchor'), got.get('err_m'))
print(f'Part 1 landings re-derived for {len(keys)} seeded targets ({n} target x pose x twin cases); disagreements {bad}; ground axes {g} (spans {np.round(spread, 1).tolist()})')
sys.exit(1 if bad else 0)
