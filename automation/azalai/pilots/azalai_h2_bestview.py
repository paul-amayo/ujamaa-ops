#!/usr/bin/env python3
"""azalai_h2_bestview.py (29 Oct 2026) — RSS Azalai Stage 1 pilot H2, on logs: for every December grow bag, the keyframe the ICRA rule
would send the robot to (FIRST sighting, the earliest keyframe whose lifted returns fell in the bag) against the keyframe the census
weights pick (BEST view, the keyframe with the most lifted returns, the HiGH census-weight analogue). Per bag and per keyframe:
camera-to-bag distance, bearing off the optical axis, lifted returns, and the SAM3 grow-bag mask the projected bag point falls in
(area in pixels, 0 if none within 25 px). Paired comparison best vs first, sign test. nerf_new python3.10:
  azalai_h2_bestview.py <stpg.json> <census dir> <dec survey root> <out.json>"""
import sys, json, glob, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import binomtest
import pycocotools.mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths
from rig_calib import load_rig
stpg, Cd, R, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
E = json.load(open(stpg))['experiences']; C = np.array(E['dec']['c2w'], float); K = load_rig(survey_paths.monos(str(R)))['intrinsics']
z = np.load(Cd / 'census_dec.npz'); B = z['bags_xyz'].astype(float)
obs = defaultdict(list)
for b, k, n in zip(z['obs_bag'], z['obs_k'], z['obs_n']): obs[int(b)].append((int(k), int(n)))
ents = {}
for f in sorted(glob.glob(str(R / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
    cdj = json.load(open(f))
    for li, fr in enumerate(cdj['frames']):
        k = int(fr['kf_idx'])
        if k not in ents: ents[k] = [x['rle'] for x in cdj['frame_entries'].get(str(li), [])]
cache = {}
def masks(k):
    if k not in cache:
        ms = []
        for rle in ents.get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool))
        if len(cache) > 300: cache.clear()
        cache[k] = ms
    return cache[k]
def view(b, k):
    p = B[b]; q = np.linalg.inv(C[k]) @ np.r_[p, 1.0]; dist = float(np.linalg.norm(C[k, :3, 3] - p))
    bear = float(np.degrees(np.arccos(np.clip(q[2] / (np.linalg.norm(q[:3]) + 1e-9), -1, 1))))
    if q[2] <= 0.3: return dict(dist=dist, bearing=bear, in_image=False, mask_area=0, n_masks=len(masks(k)))
    u, v = K['fx'] * q[0] / q[2] + K['cx'], K['fy'] * q[1] / q[2] + K['cy']
    if not (0 <= u < 1280 and 0 <= v < 720): return dict(dist=dist, bearing=bear, in_image=False, mask_area=0, n_masks=len(masks(k)))
    u0, v0 = int(u), int(v); area = 0
    for m in masks(k):
        if m[max(0, v0 - 25):v0 + 26, max(0, u0 - 25):u0 + 26].any(): area = max(area, int(m.sum()))
    return dict(dist=dist, bearing=bear, in_image=True, mask_area=area, n_masks=len(masks(k)), u=float(u), v=float(v))
rows = []
for b, l in sorted(obs.items()):
    if len(l) < 2 or not np.isfinite(B[b]).all(): continue
    kf = min(l)[0]; kb = max(l, key=lambda t: t[1])[0]; d = dict(l)
    rows.append(dict(bag=b, n_views=len(l), first=dict(k=kf, returns=d[kf], **view(b, kf)), best=dict(k=kb, returns=d[kb], **view(b, kb))))
    if len(rows) % 100 == 0: print('[h2]', len(rows), flush=True)
def paired(key, better):
    a = np.array([r['first'][key] for r in rows], float); b_ = np.array([r['best'][key] for r in rows], float)
    w = int((better(b_, a)).sum()); l = int((better(a, b_)).sum()); t = len(rows) - w - l
    return dict(first_median=float(np.median(a)), best_median=float(np.median(b_)), best_better=w, first_better=l, ties=t, sign_test_p=float(binomtest(w, w + l).pvalue) if w + l else None)
S = dict(n_bags=len(rows), same_keyframe=int(sum(r['first']['k'] == r['best']['k'] for r in rows)),
         distance_m=paired('dist', lambda x, y: x < y), bearing_deg=paired('bearing', lambda x, y: x < y), returns=paired('returns', lambda x, y: x > y),
         mask_area_px=paired('mask_area', lambda x, y: x > y),
         mask_hit_rate=dict(first=float(np.mean([r['first']['mask_area'] > 0 for r in rows])), best=float(np.mean([r['best']['mask_area'] > 0 for r in rows]))),
         in_image=dict(first=float(np.mean([r['first']['in_image'] for r in rows])), best=float(np.mean([r['best']['in_image'] for r in rows]))),
         rule=dict(first='earliest keyframe with lifted returns in the bag (ICRA first-sighting rule)', best='keyframe with the most lifted returns (census weight)', mask='SAM3 grow bag conf 0.1 mask within 25 px of the projected bag point'))
json.dump(dict(summary=S, bags=rows), open(out, 'w'), indent=1); print(json.dumps(S, indent=1))
