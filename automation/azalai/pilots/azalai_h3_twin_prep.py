#!/usr/bin/env python3
"""azalai_h3_twin_prep.py (11 Nov 2026) — RSS Azalai Stage 1 pilot H3 with the twin render, step 1 of 3: GEOMETRY ONLY, no score is
computed here. For the 552 (target bag, April goal keyframe) pairs of the positional pilot (azalai_h3_arrival.py, 29 Oct; the pairs
are read from its record, its two neighbours are recomputed with its rule and checked against its wrong keyframe) this writes what
the renderer and the scorer need:
  * frames: the December graph (STPG) and the ten twin cells share one world up to a rigid map T (Kabsch on the camera centres of
    the keyframes both carry). A point goes graph -> cell by T plus the position residual d_k of the nearest ANCHOR keyframe k (the
    nearest December keyframe of the twin with the same heading); a camera keeps its graph orientation times C_k, the rotation
    between the graph's and the cell's camera frame at k (the cells' cameras were refined with their own intrinsics; the cell
    images are the keyframe images pixel for pixel, so cell pose + cell intrinsics give image pixels). Exact at a December keyframe.
  * the BELIEVED pose of each pair in the twin: the April goal keyframe carried into the December graph by the pilot's rigid fit
    (placed_<apr>.npz G_global, as the pilot carried the footprints), then into the cell as above;
  * the twin identity of each December census bag: mutual nearest neighbour within 0.5 m in the ground plane between the census
    bags and the twin's anchors (marker_hierarchy.json objects), the pairing rule of the pilot; a supervision-vote check is printed;
  * the candidates: the target's identity and its row neighbours in the twin's own hierarchy, one each side along the row;
  * the COUNTERFACTUAL poses of rule B: the believed pose translated by (neighbour anchor - target anchor);
  * the same-season pairs: the December HELD-OUT keyframe (splat test split) with the most lifted returns on the target, and for the
    wrong arrival the held-out keyframe with the most returns on the pilot's first neighbour other than the target's own;
  * the changed / unchanged call of each target from registry v0 (growth: stable / grown / shrunk / split / none).
nerf_new python3.10:
  azalai_h3_twin_prep.py <stpg.json> <census dir> <apr name> <pilot h3 json> <registry_v0.json> <out prep.json>"""
import sys, json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from PIL import Image
stpg, Cdir, APR, pilot, registry, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
D = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental'); GL = np.diag([1.0, -1.0, -1.0, 1.0])
E = json.load(open(stpg))['experiences']; Cd = np.array(E['dec']['c2w'], float); tsd = np.array(E['dec']['ts_ms']); Ca = np.array(E[APR]['c2w'], float)
zd = np.load(Cdir / 'census_dec.npz'); dec = zd['bags_xyz'].astype(float); PL = np.load(Cdir / f'placed_{APR}.npz'); S = PL['global'].astype(float); G = PL['G_global'].astype(float)
za = np.load(Cdir / f'census_{APR}.npz')
# ---------------------------------------------------------------- the cells of the twin
inv = [json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')]; inv = [c for c in inv if c['survey'] == 'dec_2025_ten_rows']
lanes = {}; kf_lane = {}   # December keyframe -> (lane, frame name)
for c in inv:
    sup = Path(c['sup']); lane = sup.parent.parent.name; ws = Path(c['ws']); tj = json.load(open(ws / 'transforms.json')); st = json.load(open(D / lane / 'stamps.json'))
    P = {Path(f['file_path']).name: np.array(f['transform_matrix'], float) @ GL for f in tj['frames']}   # OpenCV c2w, cell world (metric)
    by_ts = {int(st[n]): n for n in P}; test = set(json.load(open(ws / 'split.json'))['test'])
    kf = {k: by_ts[int(t)] for k, t in enumerate(tsd) if int(t) in by_ts and np.isfinite(Cd[k]).all()}
    for k, n in kf.items(): kf_lane.setdefault(k, (lane, n))
    bank = {}
    for tw, idd in (('canon', Path(c['identity'])), ('bag', ws / 'identity_masks_only_v1')):
        b = np.load(idd / 'text_bank.npz', allow_pickle=True); words = set(b['words']); wt = json.loads(str(b['meta']))['word_table']
        bank[tw] = dict(identity=str(idd), ids=sorted(int(k) for k, w in wt.items() if int(k) < 10000 and w in words))
    lanes[lane] = dict(ws=str(ws), sup=str(sup), config=json.load(open(Path(c['identity']) / 'W.json'))['config'], K=[tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy']], wh=[tj['w'], tj['h']],
                       P=P, st=st, kf=kf, test=test, bank=bank, maps=sorted(p.name for p in sup.glob('f_*.png')))
    print(f"[prep] {lane}: {len(P)} frames, {len(kf)} December keyframes (k {min(kf)}..{max(kf)}), {len(test)} held-out frames; plant addresses canon {len(bank['canon']['ids'])}, bag-level {len(bank['bag']['ids'])}", flush=True)
KF = np.array(sorted(kf_lane)); KFp = Cd[KF][:, :3, 3]; KFz = Cd[KF][:, :3, 2]; Pk = {int(k): lanes[kf_lane[int(k)][0]]['P'][kf_lane[int(k)][1]] for k in KF}
# one rigid map graph -> cell world from the camera centres (the ten cells share a world)
a_, b_ = KFp, np.array([Pk[int(k)][:3, 3] for k in KF]); ma, mb = a_.mean(0), b_.mean(0); U, _, Vt = np.linalg.svd((a_ - ma).T @ (b_ - mb)); TR = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T; Tt = mb - TR @ ma
res = np.linalg.norm((TR @ a_.T).T + Tt - b_, axis=1); print(f'[prep] graph -> cell world: {len(KF)} keyframes, residual median {np.median(res):.3f} m, p90 {np.percentile(res, 90):.3f}, max {res.max():.3f}', flush=True)
dk = {int(k): Pk[int(k)][:3, 3] - (TR @ Cd[k][:3, 3] + Tt) for k in KF}; Ck = {int(k): (TR @ Cd[k][:3, :3]).T @ Pk[int(k)][:3, :3] for k in KF}
def anchor(pose):
    """nearest December keyframe of the twin with the same heading (optical axes within 90 deg) -> (k, lane, frame, distance m)"""
    d = np.linalg.norm(KFp - pose[:3, 3], axis=1); d[(KFz @ pose[:3, 2]) <= 0] = np.inf; i = int(np.argmin(d))
    return (int(KF[i]), *kf_lane[int(KF[i])], float(d[i])) if np.isfinite(d[i]) else None
def pt(p, k): return TR @ p + Tt + dk[k]                       # graph point -> cell world, anchored at keyframe k
def cam(pose, k):                                              # graph camera -> cell camera (OpenCV c2w), anchored at keyframe k
    q = np.eye(4); q[:3, :3] = TR @ pose[:3, :3] @ Ck[k]; q[:3, 3] = pt(pose[:3, 3], k); return q
assert max(np.abs(cam(Cd[int(k)], int(k)) - Pk[int(k)]).max() for k in KF[:50]) < 1e-6
# ---------------------------------------------------------------- twin identity of each December census bag (mutual nearest anchor)
Hj = json.load(open(D / 'identity/marker_hierarchy.json')); O = np.array([o['xyz'] for o in Hj['objects']], float); oid = [int(o['id']) for o in Hj['objects']]; opos = {i: O[j] for j, i in enumerate(oid)}
ra = np.array(json.load(open(D / 'identity/tree_census.json'))['row_axis'], float)
row_nb = {}
for r in Hj['rows']:
    ii = sorted(r['object_ids'], key=lambda i: float(opos[i][:2] @ ra))
    for n, i in enumerate(ii): row_nb[int(i)] = (int(ii[n - 1]) if n > 0 else None, int(ii[n + 1]) if n + 1 < len(ii) else None)
okd = np.flatnonzero(np.isfinite(dec).all(1)); Bc = (TR @ dec[okd].T).T + Tt
dO, jO = cKDTree(O[:, :2]).query(Bc[:, :2]); dB_, jB_ = cKDTree(Bc[:, :2]).query(O[:, :2])
twin_id = {int(okd[i]): oid[jO[i]] for i in range(len(okd)) if dO[i] <= 0.5 and jB_[jO[i]] == i}; twin_d = {int(okd[i]): float(dO[i]) for i in range(len(okd))}
print(f'[prep] twin identity: {len(twin_id)} of {len(okd)} December census bags are the mutual nearest neighbour of a twin anchor within 0.5 m (median distance {np.median([twin_d[b] for b in twin_id]):.3f} m); {len(O)} anchors in {len(Hj["rows"])} rows', flush=True)
# check of the naming against the supervision itself: the id at the pixel the bag point projects to, over the cell frames with a map
votes = defaultdict(Counter)
for lane, Ln in lanes.items():
    fx, fy, cx, cy = Ln['K']
    for n in Ln['maps'][::3]:
        if n not in Ln['P']: continue
        w2c = np.linalg.inv(Ln['P'][n]); q = (w2c[:3, :3] @ Bc.T).T + w2c[:3, 3]; m = (q[:, 2] > 0.5) & (q[:, 2] < 3.0)
        if not m.any(): continue
        u = fx * q[:, 0] / q[:, 2] + cx; v = fy * q[:, 1] / q[:, 2] + cy; m &= (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        if not m.any(): continue
        ids = np.array(Image.open(Path(Ln['sup']) / n))
        for b, uu, vv in zip(okd[m], u[m].astype(int), v[m].astype(int)):
            if ids[vv, uu] < 10000: votes[int(b)][int(ids[vv, uu])] += 1
vt = {b: c.most_common(1)[0][0] for b, c in votes.items() if sum(c.values()) >= 3}; both = [b for b in vt if b in twin_id]
print(f'[prep] naming check: of {len(both)} named bags with >= 3 supervision votes, the vote majority is the same identity for {sum(vt[b] == twin_id[b] for b in both)}', flush=True)
# ---------------------------------------------------------------- the pilot's pairs, its neighbours, its goal keyframes
rec = json.load(open(pilot))['targets']
oka = np.flatnonzero(np.isfinite(S).all(1)); td = cKDTree(dec[okd][:, [0, 2]]); ta = cKDTree(S[oka][:, [0, 2]])
dA, jA = ta.query(dec[okd][:, [0, 2]]); dB, jB = td.query(S[oka][:, [0, 2]])
pairs = [(int(okd[i]), int(oka[jA[i]])) for i in range(len(okd)) if dA[i] <= 0.5 and jB[jA[i]] == i]
obs = defaultdict(list)
for b, k, n in zip(za['obs_bag'], za['obs_k'], za['obs_n']): obs[int(b)].append((int(k), int(n)))
goal_of = {d_i: max(obs[a_i], key=lambda t: t[1])[0] for d_i, a_i in pairs if obs.get(a_i)}
dd, nn = td.query(dec[okd][:, [0, 2]], k=3); nbr = {int(okd[i]): [int(okd[nn[i][1]]), int(okd[nn[i][2]])] for i in range(len(okd))}
obs_d = defaultdict(list)
for b, k, n in zip(zd['obs_bag'], zd['obs_k'], zd['obs_n']): obs_d[int(b)].append((int(k), int(n)))
held = {k for k, (lane, n) in kf_lane.items() if n in lanes[lane]['test']}
reg = {b['id']: b for b in json.load(open(registry))['bags'] if isinstance(b['id'], int)}
def shifted(pose, d): q = pose.copy(); q[:3, 3] = q[:3, 3] + d; return q
out_pairs = []; jobs = []; same = []
def candidates(t, lane):
    """twin identity of the target and of its row neighbours (one each side) + which are in the cell's vocabulary (both twins)"""
    ti = twin_id.get(t); nb = row_nb.get(ti, (None, None)) if ti is not None else (None, None); c = dict(t=ti, prev=nb[0], next=nb[1])
    return c, {tw: {k_: (v is not None and v in lanes[lane]['bank'][tw]['ids']) for k_, v in c.items()} for tw in ('canon', 'bag')}
def vote(t): return 'none' if t not in twin_id else ('no votes' if t not in vt else ('agree' if vt[t] == twin_id[t] else 'disagree'))
def add_jobs(prefix, t, lane, pose, c):
    for tag, o in (('V0', None), ('Vprev', c['prev']), ('Vnext', c['next'])):
        if tag != 'V0' and o is None: continue
        jobs.append(dict(job=f'{prefix}_{t}_{tag}', lane=lane, c2w=(pose if o is None else shifted(pose, opos[o] - opos[c['t']])).tolist(), rgb=(tag == 'V0')))
for r in rec:
    t, g, gw = r['dec'], r['goal_k'], r['wrong_k']; n1, n2 = nbr[t]
    assert goal_of[t] == g and goal_of[n1] == gw, (t, g, gw, goal_of[t], goal_of.get(n1))
    e = dict(dec=t, apr=r['apr'], n1=n1, n2=n2, goal_k=g, wrong_k=gw, n1_twin=twin_id.get(n1), n1_offset=dict(along_m=float(dec[n1][2] - dec[t][2]), across_m=float(dec[n1][0] - dec[t][0])),
             growth=(reg.get(t) or {}).get('growth'), status=(reg.get(t) or {}).get('status'), naming_vote=vote(t), positional=dict(correct=r['correct']['arrived'], wrong=r['wrong']['arrived']))
    Pb = G @ Ca[g]; a = anchor(Pb) if np.isfinite(Pb).all() else None
    if a is None: e['skip'] = 'no December keyframe of the twin with the same heading'; out_pairs.append(e); same.append(dict(dec=t, skip=e['skip'])); continue
    k, lane, n, dist = a; pose = cam(Pb, k); c, inb = candidates(t, lane)
    e.update(lane=lane, anchor_k=k, anchor_frame=n, anchor_dist_m=dist, twin=c, in_bank=inb, target_range_m=float(np.linalg.norm(pt(dec[t], k) - pose[:3, 3])),
             n1_is_candidate=(e['n1_twin'] is not None and e['n1_twin'] in (c['prev'], c['next'])),
             true_wrong_displacement_m=float(np.linalg.norm((G @ Ca[gw])[:3, 3] - Pb[:3, 3])))   # for the record only: never rendered, never given to the check
    if c['t'] is None: e['skip'] = 'target has no identity in the twin'
    elif not inb['canon']['t']: e['skip'] = "target's identity is not in the cell's vocabulary"
    else: add_jobs('apr', t, lane, pose, c)
    out_pairs.append(e)
    # same season: December held-out keyframes
    hs = [(k_, n_) for k_, n_ in obs_d.get(t, []) if k_ in held]; s = dict(dec=t, n1=n1, growth=e['growth'], n1_twin=e['n1_twin'], naming_vote=e['naming_vote'])
    if not hs: s['skip'] = 'no held-out December keyframe sees the target'; same.append(s); continue
    gd = max(hs, key=lambda x: x[1])[0]; hn = [(k_, n_) for k_, n_ in obs_d.get(n1, []) if k_ in held and k_ != gd]
    if not hn: s['skip'] = 'no other held-out December keyframe sees the neighbour'; same.append(s); continue
    gdw = max(hn, key=lambda x: x[1])[0]; lane_d, nd = kf_lane[gd]; pd = Pk[gd]; c, inb = candidates(t, lane_d)
    s.update(goal_k=gd, wrong_k=gdw, lane=lane_d, frame=nd, wrong_frame=kf_lane[gdw][1], twin=c, in_bank=inb, target_range_m=float(np.linalg.norm(pt(dec[t], gd) - pd[:3, 3])),
             n1_is_candidate=(s['n1_twin'] is not None and s['n1_twin'] in (c['prev'], c['next'])), true_wrong_displacement_m=float(np.linalg.norm(Cd[gdw][:3, 3] - Cd[gd][:3, 3])))
    if c['t'] is None: s['skip'] = 'target has no identity in the twin'
    elif not inb['canon']['t']: s['skip'] = "target's identity is not in the cell's vocabulary"
    else: add_jobs('dec', t, lane_d, pd, c)
    same.append(s)
def cover(L):
    ok = [p for p in L if 'skip' not in p]
    return dict(evaluable=len(ok), skipped=dict(Counter(p['skip'] for p in L if 'skip' in p)), two_neighbours=sum(p['twin']['prev'] is not None and p['twin']['next'] is not None for p in ok),
                neighbours_in_cell=sum(all(p['in_bank']['canon'][c] for c in ('prev', 'next') if p['twin'][c] is not None) for p in ok), n1_is_candidate=sum(p['n1_is_candidate'] for p in ok), naming_vote=dict(Counter(p['naming_vote'] for p in ok)),
                growth=dict(Counter(str(p['growth']) for p in ok)), target_range_m=float(np.median([p['target_range_m'] for p in ok])), true_wrong_displacement_m=float(np.median([p['true_wrong_displacement_m'] for p in ok])))
po = [p for p in out_pairs if 'anchor_dist_m' in p]
summ = dict(n_pairs=len(out_pairs), april=cover(out_pairs), december_held_out=cover(same), anchor_dist_m=dict(median=float(np.median([p['anchor_dist_m'] for p in po])), p90=float(np.percentile([p['anchor_dist_m'] for p in po], 90))),
            growth_all=dict(Counter(str(p['growth']) for p in out_pairs)), graph_to_cell_resid_m=dict(median=float(np.median(res)), p90=float(np.percentile(res, 90))),
            global_fit_resid_med_m=float(PL['global_fit_resid_med']), twin_named=len(twin_id), jobs=len(jobs))
json.dump(dict(summary=summ, apr=APR, G_global=G.tolist(), T_graph_to_cell=dict(R=TR.tolist(), t=Tt.tolist()),
               lanes={l: dict(ws=v['ws'], sup=v['sup'], config=v['config'], K=v['K'], wh=v['wh'], bank=v['bank']) for l, v in lanes.items()},
               twin_id={str(k): v for k, v in twin_id.items()}, anchors={str(i): opos[i].tolist() for i in oid}, pairs=out_pairs, same_season=same, jobs=jobs), open(out, 'w'))
print(json.dumps(summ, indent=1))
