#!/usr/bin/env python3
"""azalai_h3_arrival.py (29 Oct 2026) — RSS Azalai Stage 1 pilot H3, on logs: the arrival check without a robot. For each December bag
found again in an April pass (mutual nearest neighbour within 0.5 m on the global route), the GOAL is the April keyframe that views its
April partner best (most lifted returns). The twin's rendered target mask is stood in for by the target's footprint (a disc of radius
FOOT m about the December bag point, carried into April by the inverse rigid fit) projected at the GOAL pose; the live frame is the
goal keyframe itself (a correct arrival) or the goal keyframe of the target's nearest row neighbour (a wrong-plant arrival, the robot
one bag along). The check: among the live frame's SAM3 masks, the one that best overlaps the projected target footprint; arrived if
that mask's IoU with the target footprint exceeds its IoU with the footprints of the two nearest neighbours projected at the same
goal pose (the argmax rule, no threshold). Reports the confusion over correct and wrong arrivals, plus the path-RMSE stand-in
(camera displacement between the two keyframes). nerf_new python3.10:
  azalai_h3_arrival.py <stpg.json> <census dir> <apr name> <apr survey root> <out.json> [--foot 0.3]"""
import sys, json, glob, argparse, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree
import pycocotools.mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths
from rig_calib import load_rig
ap = argparse.ArgumentParser(); ap.add_argument('stpg'); ap.add_argument('census'); ap.add_argument('apr'); ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--foot', type=float, default=0.3); a = ap.parse_args()
Cd = Path(a.census); R = Path(a.root); E = json.load(open(a.stpg))['experiences']; C = np.array(E[a.apr]['c2w'], float); K = load_rig(survey_paths.monos(str(R)))['intrinsics']
dec = np.load(Cd / 'census_dec.npz')['bags_xyz'].astype(float); PL = np.load(Cd / f'placed_{a.apr}.npz'); S = PL['global'].astype(float); M = np.linalg.inv(PL['G_global'])
za = np.load(Cd / f'census_{a.apr}.npz'); A = za['bags_xyz'].astype(float)
obs = defaultdict(list)
for b, k, n in zip(za['obs_bag'], za['obs_k'], za['obs_n']): obs[int(b)].append((int(k), int(n)))
okd = np.flatnonzero(np.isfinite(dec).all(1)); oka = np.flatnonzero(np.isfinite(S).all(1))
td = cKDTree(dec[okd][:, [0, 2]]); ta = cKDTree(S[oka][:, [0, 2]])
dA, jA = ta.query(dec[okd][:, [0, 2]]); dB, jB = td.query(S[oka][:, [0, 2]])
pairs = [(int(okd[i]), int(oka[jA[i]])) for i in range(len(okd)) if dA[i] <= 0.5 and jB[jA[i]] == i]
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
        if len(cache) > 200: cache.clear()
        cache[k] = ms
    return cache[k]
vv, uu = np.mgrid[0:720, 0:1280]
def footprint(p_dec, k):
    """disc of radius foot about the bag point (in the April frame) on the ground plane, projected into keyframe k -> boolean image"""
    p = (M @ np.r_[p_dec, 1.0])[:3]; w2c = np.linalg.inv(C[k]); img = np.zeros((720, 1280), bool)
    # sample the disc (horizontal plane through the bag point) and the bag's vertical extent 0..0.5 m, convex hull by projection of samples
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False); pts = []
    for h in (0.0, 0.25, 0.5):
        for t in ang: pts.append(p + np.array([a.foot * np.cos(t), -h, a.foot * np.sin(t)]))
    q = (w2c[:3, :3] @ np.array(pts).T).T + w2c[:3, 3]; q = q[q[:, 2] > 0.3]
    if len(q) < 6: return img
    u = K['fx'] * q[:, 0] / q[:, 2] + K['cx']; v = K['fy'] * q[:, 1] / q[:, 2] + K['cy']
    if (u.max() < 0) or (u.min() >= 1280) or (v.max() < 0) or (v.min() >= 720): return img
    from scipy.spatial import ConvexHull, Delaunay
    P = np.c_[u, v]
    try: hull = Delaunay(P[ConvexHull(P).vertices])
    except Exception: return img
    u0, u1 = int(max(0, u.min())), int(min(1279, u.max())); v0, v1 = int(max(0, v.min())), int(min(719, v.max()))
    sub = np.c_[uu[v0:v1 + 1, u0:u1 + 1].ravel(), vv[v0:v1 + 1, u0:u1 + 1].ravel()]
    img[v0:v1 + 1, u0:u1 + 1] = (hull.find_simplex(sub) >= 0).reshape(v1 - v0 + 1, u1 - u0 + 1); return img
def iou(x, y):
    u = (x | y).sum(); return float((x & y).sum() / u) if u else 0.0
def check(goal_k, live_k, p_t, p_n1, p_n2):
    ft, f1, f2 = footprint(p_t, goal_k), footprint(p_n1, goal_k), footprint(p_n2, goal_k)
    if not ft.any(): return None
    ms = masks(live_k)
    if not ms: return dict(arrived=False, reason='no live masks', iou_t=0.0, iou_n=0.0)
    best = max(ms, key=lambda m: iou(m, ft)); it = iou(best, ft); i_n = max(iou(best, f1), iou(best, f2))
    return dict(arrived=bool(it > 0 and it > i_n), iou_t=it, iou_n=i_n, n_masks=len(ms))
res = []; goal_of = {}
for d_i, a_i in pairs:
    if obs.get(a_i): goal_of[d_i] = max(obs[a_i], key=lambda t: t[1])[0]
dd, nn = td.query(dec[okd][:, [0, 2]], k=3)
nbr = {int(okd[i]): [int(okd[nn[i][1]]), int(okd[nn[i][2]])] for i in range(len(okd))}
for d_i, a_i in pairs:
    if d_i not in goal_of: continue
    n1, n2 = nbr[d_i]
    if n1 not in goal_of: continue
    g = goal_of[d_i]; gw = goal_of[n1]
    if g == gw: continue
    ok_ = check(g, g, dec[d_i], dec[n1], dec[n2]); wr = check(g, gw, dec[d_i], dec[n1], dec[n2])
    if ok_ is None or wr is None: continue
    disp = float(np.linalg.norm(C[g, :3, 3] - C[gw, :3, 3]))
    res.append(dict(dec=d_i, apr=a_i, goal_k=g, wrong_k=gw, neighbour_dist_m=float(dd[list(okd).index(d_i)][1]), displacement_m=disp, correct=ok_, wrong=wr))
    if len(res) % 100 == 0: print('[h3]', len(res), flush=True)
tp = sum(r['correct']['arrived'] for r in res); fn = len(res) - tp; fa = sum(r['wrong']['arrived'] for r in res); tn = len(res) - fa
S = dict(n_targets=len(res), correct_arrival=dict(confirmed=tp, rejected=fn), wrong_arrival=dict(confirmed_false_alarm=fa, rejected=tn),
         displacement_m=dict(median=float(np.median([r['displacement_m'] for r in res])), p10=float(np.percentile([r['displacement_m'] for r in res], 10)), p90=float(np.percentile([r['displacement_m'] for r in res], 90))),
         neighbour_dist_m=dict(median=float(np.median([r['neighbour_dist_m'] for r in res]))),
         iou=dict(correct_target_median=float(np.median([r['correct']['iou_t'] for r in res])), wrong_target_median=float(np.median([r['wrong']['iou_t'] for r in res]))),
         rule='live mask best overlapping the target footprint projected at the goal pose; arrived if its IoU with the target footprint beats its IoU with the two nearest neighbours (argmax, no threshold)', foot_m=a.foot, pass_=a.apr)
json.dump(dict(summary=S, targets=res), open(a.out, 'w'), indent=1); print(json.dumps(S, indent=1))
