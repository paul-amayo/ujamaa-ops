#!/usr/bin/env python3
"""twin_ledger_prep.py (11 Nov 2026) — JIRS test JS-1 (the ledger read through the twin's identity field), step 1 of 4:
GEOMETRY AND NAMING ONLY, no identity feature is read here. Design of record: Sankofa_JIRS/numbers/twin_ledger/DESIGN.md.
For the December twin (ten lane cells, one world) and the April 10:54 twin (four chunk cells, one world) this writes what the
renderer and the scorer need:
  * the frames of each cell that are training frames AND keyframes of the pass's graph (December by time stamp, April chunks
    by keyframe index), with the cell pose (OpenCV c2w, metric cell world);
  * one rigid map graph -> cell world per twin (Kabsch on the camera centres of the keyframes both carry). A point x seen at
    keyframe k goes to the graph frame as c_graph(k) + R^T (x - c_cell(k)) (camera centres only);
  * the plant addresses of each cell (ids below 10,000 of the cell's text bank);
  * the naming: which census bag bears which identity = mutual nearest neighbour within 0.5 m on the ground plane between the
    census bags and the twin's anchors (the rule of azalai_h3_twin_prep.py). December: bags carried into the cell world by the
    rigid map, anchors' x/y. April: anchors are in the pass's odometry frame, compared directly in x/z;
  * the check of the naming against the supervision: the identity under the pixel each census bag projects to in the cells'
    supervision maps (keyframes, bag carried through the keyframe's own centre), majority over frames with >= 3 votes.
The April lane cell (chunk_lane) is not used (design: its fit to the graph leaves the roll undetermined). nerf_new python3.10:
  twin_ledger_prep.py <out prep.json>"""
import sys, json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from PIL import Image
out = sys.argv[1]; GL = np.diag([1.0, -1.0, -1.0, 1.0]); RAD = 0.5
X = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental'); A = Path('/home/paperspace/data/klapmuts/apr_2026_zed')
L = X / 'sankofa/jirs_20261023/lidar_fixedreader'; STPG = X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'
E = json.load(open(STPG))['experiences']; inv = [json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')]
def kabsch(a_, b_):
    ma, mb = a_.mean(0), b_.mean(0); U, _, Vt = np.linalg.svd((a_ - ma).T @ (b_ - mb)); R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T; return R, mb - R @ ma
def mutual(P, Q, ip, iq):
    """mutual nearest neighbours within RAD between point sets P (ids ip) and Q (ids iq) -> {ip: (iq, d)}"""
    dP, jP = cKDTree(Q).query(P); dQ, jQ = cKDTree(P).query(Q); return {int(ip[i]): (int(iq[jP[i]]), float(dP[i])) for i in range(len(P)) if dP[i] <= RAD and jQ[jP[i]] == i}
def bank_ids(idd):
    b = np.load(Path(idd) / 'text_bank.npz', allow_pickle=True); words = set(b['words']); wt = json.loads(str(b['meta']))['word_table']
    return sorted(int(k) for k, w in wt.items() if int(k) < 10000 and w in words)
P_out = {}
for season, survey, pas in (('dec', 'dec_2025_ten_rows', 'dec'), ('apr', 'apr_2026_zed', 'apr1054')):
    Cg = np.array(E[pas]['c2w'], float); ts = np.array(E[pas]['ts_ms']); cells = {}; kfP = {}; left_out = []
    for c in [c for c in inv if c['survey'] == survey]:
        ws = Path(c['ws']); name = c['chunk']
        if season == 'apr' and name == 'chunk_lane': left_out.append(dict(cell=name, reason='own world on one straight path: roll of the fit to the graph undetermined (design)')); continue
        tj = json.load(open(ws / 'transforms.json')); sp = json.load(open(ws / 'split.json')); train = set(sp['train']); assert all(abs(float(tj.get(k_, 0.0))) < 1e-9 for k_ in ('k1', 'k2', 'p1', 'p2')), f'{name}: distortion'
        P = {Path(f['file_path']).name: np.array(f['transform_matrix'], float) @ GL for f in tj['frames']}
        if season == 'dec':
            name = Path(c['sup']).parent.parent.name; st = json.load(open(X / name / 'stamps.json')); by_ts = {int(st[n]): n for n in P}
            kf = {k: by_ts[int(t)] for k, t in enumerate(ts) if int(t) in by_ts and np.isfinite(Cg[k]).all()}
        else: kf = {int(n[3:9]): n for n in P if int(n[3:9]) < len(Cg) and np.isfinite(Cg[int(n[3:9])]).all()}
        for k, n in kf.items(): kfP.setdefault(k, P[n])
        sup = Path(c['sup']); maps = set(p.name for p in sup.glob('*.png')); ids = bank_ids(c['identity'])
        cells[name] = dict(ws=str(ws), identity=c['identity'], sup=str(sup), config=json.load(open(Path(c['identity']) / 'W.json'))['config'], K=[tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy']], wh=[tj['w'], tj['h']], ids=ids,
                           frames=[dict(k=int(k), name=n, c2w=P[n].tolist(), map=(n in maps)) for k, n in sorted(kf.items()) if n in train], n_frames=len(P), n_keyframes=len(kf), n_heldout_keyframes=sum(n not in train for n in kf.values()), _P=P, _kf=kf, _maps=maps)
        print(f"[prep] {season} {name}: {len(P)} frames, {len(kf)} graph keyframes, {len(cells[name]['frames'])} of them training frames (used), {cells[name]['n_heldout_keyframes']} held out; {len(ids)} plant addresses", flush=True)
    KF = np.array(sorted(kfP)); a_ = Cg[KF][:, :3, 3]; b_ = np.array([kfP[int(k)][:3, 3] for k in KF]); R, t = kabsch(a_, b_); res = np.linalg.norm(a_ @ R.T + t - b_, axis=1)
    print(f'[prep] {season}: graph -> cell world on {len(KF)} keyframes, residual median {np.median(res):.3f} m, p90 {np.percentile(res, 90):.3f}, max {res.max():.3f}', flush=True)
    # ---- naming
    z = np.load(L / f'census_{pas}.npz'); B = z['bags_xyz'].astype(float); okb = np.flatnonzero(np.isfinite(B).all(1))
    if season == 'dec':
        Hj = json.load(open(X / 'identity/marker_hierarchy.json')); O = np.array([o['xyz'] for o in Hj['objects']], float); oid = np.array([int(o['id']) for o in Hj['objects']]); m = mutual((B[okb] @ R.T + t)[:, :2], O[:, :2], okb, oid)
        anchors_graph = (O - t) @ R   # R^T (O - t): anchors in the graph frame (rigid), for the record
    else:
        Hj = json.load(open(A / 'prod/bateleur/scene_graph/marker_hierarchy.json')); O = np.array([o['xyz'] for o in Hj['objects']], float); oid = np.array([int(o['id']) for o in Hj['objects']]); m = mutual(B[okb][:, [0, 2]], O[:, [0, 2]], okb, oid)
        anchors_graph = O
    vocab = sorted(set(i for c in cells.values() for i in c['ids'])); invoc = {b: (i in set(vocab)) for b, (i, d) in m.items()}
    print(f'[prep] {season}: {len(m)} of {len(okb)} census bags are the mutual nearest neighbour of a twin anchor within {RAD} m (median distance {np.median([d for _, d in m.values()]):.3f} m); {len(O)} anchors; vocabulary {len(vocab)} addresses over the used cells; named bags with the identity in the vocabulary {sum(invoc.values())}', flush=True)
    # ---- naming check against the supervision maps (keyframes with a map; bag carried through the keyframe's own centre)
    votes = defaultdict(Counter)
    for name, c in cells.items():
        fx, fy, cx, cy = c['K']; W_, H_ = c['wh']
        for k, n in sorted(c['_kf'].items())[::2]:
            if n not in c['_maps']: continue
            Pc = c['_P'][n]; q = (B[okb] - Cg[k][:3, 3]) @ R.T @ Pc[:3, :3]   # camera-frame coordinates: P^T R (p - c_g)
            mm = (q[:, 2] > 0.5) & (q[:, 2] < 3.0)
            if not mm.any(): continue
            u = fx * q[:, 0] / q[:, 2] + cx; v = fy * q[:, 1] / q[:, 2] + cy; mm &= (u >= 0) & (u < W_) & (v >= 0) & (v < H_)
            if not mm.any(): continue
            ids_ = np.array(Image.open(Path(c['sup']) / n))
            if ids_.ndim == 3: raise SystemExit(f'{name}: supervision map {n} is not an id map: {ids_.shape}')
            for b, uu, vv in zip(okb[mm], u[mm].astype(int), v[mm].astype(int)):
                if ids_[vv, uu] < 10000: votes[int(b)][int(ids_[vv, uu])] += 1
    vt = {b: c_.most_common(1)[0][0] for b, c_ in votes.items() if sum(c_.values()) >= 3}; both = [b for b in vt if b in m]
    print(f'[prep] {season}: naming check: of {len(both)} named bags with >= 3 supervision votes, the vote majority is the same identity for {sum(vt[b] == m[b][0] for b in both)}; bags with a vote majority {len(vt)}', flush=True)
    hz = np.load(L / f'heights_{pas}.npz')
    for c in cells.values():
        for k_ in ('_P', '_kf', '_maps'): c.pop(k_)
    P_out[season] = dict(survey=survey, pas=pas, cells=cells, left_out=left_out, R=R.tolist(), t=t.tolist(), fit=dict(n=len(KF), resid_median_m=float(np.median(res)), resid_p90_m=float(np.percentile(res, 90))),
                         cam_centres_graph={str(int(k)): Cg[int(k)][:3, 3].tolist() for k in KF}, up=hz['up'].tolist(), mount_lane_lidar=float(hz['mount_lane']), vocab=vocab,
                         naming={str(b): dict(identity=i, anchor_dist_m=d, in_vocab=bool(invoc[b]), vote=(vt.get(b)), vote_agrees=(None if b not in vt else bool(vt[b] == i))) for b, (i, d) in m.items()},
                         vote_only={str(b): v for b, v in vt.items() if b not in m}, anchors_graph={str(int(i)): anchors_graph[j].tolist() for j, i in enumerate(oid)}, n_census=int(len(okb)), n_anchors=int(len(O)))
P_out['G_global'] = np.load(L / 'placed_apr1054.npz')['G_global'].tolist(); P_out['stpg'] = str(STPG); P_out['census_dir'] = str(L)
json.dump(P_out, open(out, 'w')); print('[prep] wrote', out)
