#!/usr/bin/env python3
"""azalai_online_resim.py (A100 agent, 11 Nov 2026): AZ2 online pilot, an independent re-implementation of the simulation from
the text of ONLINE_design.md (5d76dfe) and the RR design's definitions, written by the agent that took the test over without
reusing any function of azalai_online_score.py or azalai_online_verify.py. It checks the simulation, not the instancing: the
causal bag lists of azalai_online_prep.py are its input.

For one (registry, revisit) pair with site-wide truth it rebuilds: lane passes (heading rule), inner rows (1.5 m from the
camera line), lane frames (plane, two parallel lines, midway, entry-end extreme bag), driving stations, the bag-to-lane
assignment, the standoff d and the spacing S; the causal axis, up and right of travel per keyframe; variants a (dead
reckoning), M (matching, plain laser odometry), b (graph poses), cp / cm (start one spacing ahead / behind) and the oracle;
station errors per keyframe; the stop of every target and its arrival and outcome. It compares them with
online_curves_<tag>.json and online_targets_<tag>.jsonl and prints every disagreement.
  azalai_online_resim.py --reg dec --rev apr1130 --records <dir> --causal <dir> --out <dir>"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for k in ('reg', 'rev', 'records', 'causal', 'out'): ap.add_argument('--' + k, required=True)
A = ap.parse_args()
TAG = f'{A.reg}_{A.rev}'
X = Path('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental')
SK = X / 'sankofa/jirs_20261023/lidar_fixedreader'
EXP = json.load(open(X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'))['experiences']
SUM = json.load(open(Path(A.records) / f'online_summary_{TAG}.json'))
REC = [json.loads(l) for l in open(Path(A.records) / f'online_targets_{TAG}.jsonl')]
CUR = {l['lane']: l for l in json.load(open(Path(A.records) / f'online_curves_{TAG}.json'))['lanes']}
issues = []


# ------------------------------------------------------------------ the passes, from the design text
def lane_passes(c2w):
    n = len(c2w); fin = np.isfinite(c2w.reshape(n, -1)).all(1); zc = c2w[:, 2, 2]
    sense = np.where(fin & (zc > 0.8), 1, np.where(fin & (zc < -0.8), -1, 0))
    runs, start = [], None
    for k in range(n):
        if sense[k] == 0:
            if start is not None: runs.append((start, k - 1)); start = None
            continue
        if start is None: start = k
        elif sense[k] != sense[start] or abs(c2w[k, 0, 3] - c2w[k - 1, 0, 3]) > 0.6: runs.append((start, k - 1)); start = k
    if start is not None: runs.append((start, n - 1))
    return [(a, b, int(sense[a])) for a, b in runs if b - a + 1 >= 30]


def load_pass(name):
    c2w = np.array(EXP[name]['c2w'], float); z = np.load(SK / f'census_{name}.npz')
    B = z['bags_xyz'].astype(float); ob, ok_, on = z['obs_bag'].astype(int), z['obs_k'].astype(int), z['obs_n'].astype(int)
    lanes = []
    for k0, k1, s in lane_passes(c2w):
        P = c2w[k0:k1 + 1, :3, 3][:, [0, 2]]; m = P.mean(0); w, V = np.linalg.eigh(np.cov((P - m).T)); d = V[:, np.argmax(w)]
        d = d if d[1] > 0 else -d; normal = np.array([d[1], -d[0]])                       # +x side for a +z line
        seen = np.unique(ob[(ok_ >= k0) & (ok_ <= k1)]); seen = seen[np.isfinite(B[seen]).all(1)]
        lat = (B[seen][:, [0, 2]] - m) @ normal; keep = np.abs(lat) <= 1.5
        ids, lat = seen[keep], lat[keep]; left = lat < 0
        lane = dict(k0=k0, k1=k1, sense=s, ids=ids, lat=lat, left=left, frame=None)
        if left.sum() >= 2 and (~left).sum() >= 2:
            Q = B[ids]; c = Q.mean(0); w3, V3 = np.linalg.eigh(np.cov((Q - c).T)); nrm = V3[:, 0]; nrm = nrm if nrm[1] < 0 else -nrm
            ez = np.array([0.0, 0.0, 1.0]); ea = ez - nrm * (nrm @ ez); ea /= np.linalg.norm(ea); el = np.cross(nrm, ea)
            u, wa = (Q - c) @ el, (Q - c) @ ea
            M_ = np.column_stack([wa, left.astype(float), (~left).astype(float)]); slope, cl, cr = np.linalg.lstsq(M_, u, rcond=None)[0]
            axis = ea + slope * el; axis /= np.linalg.norm(axis); mid = c + 0.5 * (cl + cr) * el
            st = (Q - mid) @ axis; origin = mid + (st.min() if s > 0 else st.max()) * axis
            lane['frame'] = (origin, axis); lane['sig'] = s * ((Q - origin) @ axis)
        lanes.append(lane)
    obs = {}
    for b, k, n in zip(ob, ok_, on): obs.setdefault(int(b), []).append((int(k), int(n)))
    return dict(c2w=c2w, B=B, lanes=lanes, obs=obs)


def dstation(lane, p):
    o, a = lane['frame']; return lane['sense'] * ((np.asarray(p, float) - o) @ a)


def placed(name):
    return np.eye(4) if name == 'dec' else np.load(SK / f'placed_{name}.npz')['G_global'].astype(float)


R_, V_ = load_pass(A.reg), load_pass(A.rev)
assert len(R_['lanes']) == len(V_['lanes']) and [l['sense'] for l in R_['lanes']] == [l['sense'] for l in V_['lanes']]
assert [[l['k0'], l['k1'], l['sense']] for l in V_['lanes']] == [[l['k0'], l['k1'], l['sense']] for l in SUM['lanes']]
# bag -> lane: the lane whose camera line is nearer
owner = {}
for i, l in enumerate(R_['lanes']):
    for b, la in zip(l['ids'], l['lat']):
        if int(b) not in owner or abs(la) < owner[int(b)][1]: owner[int(b)] = (i, abs(float(la)))
dist = []
for b, (i, _) in owner.items():
    l = R_['lanes'][i]
    if l['frame'] is None: continue
    kb = max(R_['obs'][b], key=lambda t: t[1])[0]; cam = R_['c2w'][kb]
    if not np.isfinite(cam).all(): continue
    dist.append(abs(dstation(l, R_['B'][b]) - dstation(l, cam[:3, 3])))
D = float(np.median(dist))
gaps = [g for l in R_['lanes'] if l['frame'] is not None for sd in (True, False) for g in np.diff(np.sort(l['sig'][l['left'] == sd]))]
S = float(np.median(gaps)); HALF = S / 2
for name, mine, theirs in (('standoff_d', D, SUM['standoff_d_m']), ('spacing', S, SUM['spacing_m'])):
    if abs(mine - theirs) > 1e-9: issues.append((name, mine, theirs))
print(f'[resim] {TAG}: d = {D:.4f} (summary {SUM["standoff_d_m"]:.4f}), S = {S:.4f} (summary {SUM["spacing_m"]:.4f}), {len(dist)} bags for d')
G = np.linalg.inv(placed(A.reg)) @ placed(A.rev)


# ------------------------------------------------------------------ the robot
_npz = {}
def causal_steps(src, i):
    f = Path(A.causal) / f'causal_{A.rev}_{src}.npz'
    if not f.exists(): return None
    if src not in _npz:
        zz = np.load(f, allow_pickle=True); _npz[src] = {k: zz[k] for k in ('lane_ok', 'W', 'step_lane', 'step_k', 'step_start', 'step_n', 'bags_xyz')}
    z = _npz[src]
    if not bool(z['lane_ok'][i]): return None
    l = V_['lanes'][i]; k0, k1, s = l['k0'], l['k1'], l['sense']; W = z['W']; P = W[k0:k1 + 1, :3, 3]; Ydn = W[k0:k1 + 1, :3, 1]
    sel = np.flatnonzero(z['step_lane'] == i); steps = []
    for j in range(k1 - k0 + 1):
        st = sel[j]; assert int(z['step_k'][st]) == k0 + j
        if np.linalg.norm(P[j] - P[0]) < 1.0: steps.append(None); continue
        Q = P[: j + 1]; q = Q.mean(0); w, V = np.linalg.eigh((Q - q).T @ (Q - q)); ax = V[:, np.argmax(w)]
        if ax @ (P[j] - P[0]) < 0: ax = -ax
        up = -Ydn[: j + 1].mean(0); up = up - (up @ ax) * ax; up /= np.linalg.norm(up); right = np.cross(ax, up)
        Xb = z['bags_xyz'][int(z['step_start'][st]): int(z['step_start'][st]) + int(z['step_n'][st])]
        o = (Xb - P[0]) @ ax; lat = (Xb - q) @ right; inn = np.abs(lat) <= 1.5
        left = (lat[inn] < 0) if s > 0 else (lat[inn] > 0)
        steps.append((float((P[j] - P[0]) @ ax), o[inn], left))
    return steps


def match(delta, o, left, sig, sleft):
    for _ in range(20):
        diffs = []
        for sd in (True, False):
            x = sig[sleft == sd]; y = o[left == sd] + delta
            if len(x) == 0 or len(y) == 0: continue
            for p in range(len(x)):
                q = int(np.argmin(np.abs(y - x[p])))
                if int(np.argmin(np.abs(x - y[q]))) == p and abs(x[p] - y[q]) <= 0.45: diffs.append(x[p] - y[q])
        if not diffs: break
        mv = float(np.median(diffs)); delta += mv
        if abs(mv) < 1e-3: break
    return delta


def drive(steps, mode, sig, sleft, shift):
    n = len(steps); before = np.full(n, np.nan); after = np.full(n, np.nan); delta = None
    for j, g in enumerate(steps):
        if g is None: continue
        orob, o, left = g; old = delta
        if mode == 'a':
            if len(o): delta = -float(o.min())
        else:
            if delta is None and len(o): delta = -float(o.min()) + shift
            if delta is not None: delta = match(delta, o, left, sig, sleft)
        if old is not None: before[j] = orob + old
        if delta is not None: after[j] = orob + delta
    return before, after


VARS = dict(M=('lo', 'm', 0.0), a=('lo', 'a', 0.0), b=('graph', 'm', 0.0), cp=('lo', 'm', 1.0), cm=('lo', 'm', -1.0))
sims = {}
for i, (lt, lr) in enumerate(zip(R_['lanes'], V_['lanes'])):
    if lt['frame'] is None or lr['frame'] is None: sims[i] = None; continue
    Pg = V_['c2w'][lr['k0']:lr['k1'] + 1, :3, 3]; truth = np.array([dstation(lt, G[:3, :3] @ p + G[:3, 3]) for p in Pg])
    sim = dict(Pg=Pg, truth=truth, var={'O': (truth.copy(), truth.copy())}); cache = {}
    for v, (src, mode, sh) in VARS.items():
        if src not in cache: cache[src] = causal_steps(src, i)
        sim['var'][v] = None if cache[src] is None else drive(cache[src], mode, lt['sig'], lt['left'], sh * S)
    sims[i] = sim
    cur = CUR.get(i + 1)
    for v in list(VARS) + ['O']:
        if sim['var'][v] is None:
            if cur is not None and cur['var'].get(v) is not None: issues.append((f'lane{i + 1}.{v}', 'no simulation here', 'curve present'))
            continue
        e = sim['var'][v][1] - truth; ce = np.array([np.nan if x is None else x for x in cur['var'][v]['err']], float)
        if not np.array_equal(np.isfinite(e), np.isfinite(ce)): issues.append((f'lane{i + 1}.{v}.belief_mask', int(np.isfinite(e).sum()), int(np.isfinite(ce).sum()))); continue
        dmax = float(np.nanmax(np.abs(e - ce))) if np.isfinite(e).any() else 0.0
        if dmax > 2e-4: issues.append((f'lane{i + 1}.{v}.station_error_maxdiff_m', dmax, 0.0))


# ------------------------------------------------------------------ the targets
def stop_point(before, after, T):
    fin = np.flatnonzero(np.isfinite(after))
    if len(fin) == 0: return None
    j0 = int(fin[0])
    if after[j0] >= T: return j0, 1.0
    for j in range(j0 + 1, len(after)):
        if np.isfinite(before[j]) and before[j] >= T: return j, float((T - after[j - 1]) / (before[j] - after[j - 1]))
        if after[j] >= T: return j, 1.0
    return None


counts = {v: {'right': 0, 'neighbour': 0, 'other': 0, 'n': 0} for v in list(VARS) + ['O']}; checked = 0
for r in REC:
    t, p = r['registry_bag'], r['partner']
    if t not in owner:
        if 'skip' not in r: issues.append((f'target{t}', 'not in an inner row', r.get('lane')))
        continue
    i = owner[t][0]; lt, lr = R_['lanes'][i], V_['lanes'][i]
    if 'skip' in r and r['skip'] == 'target does not stand in an inner row of a lane': issues.append((f'target{t}', f'inner row of lane {i + 1}', r['skip'])); continue
    if lt['frame'] is None or lr['frame'] is None or p not in set(lr['ids'].tolist()):
        if 'skip' not in r: issues.append((f'target{t}', 'not a target here', 'target in records'))
        continue
    jt = int(np.flatnonzero(lt['ids'] == t)[0]); jp = int(np.flatnonzero(lr['ids'] == p)[0]); side = bool(lt['left'][jt])
    if bool(lr['left'][jp]) != side:
        if 'skip' not in r: issues.append((f'target{t}', 'sides differ', 'target in records'))
        continue
    if 'skip' in r: issues.append((f'target{t}', 'a target here', r['skip'])); continue
    if r['lane'] != i + 1: issues.append((f'target{t}.lane', i + 1, r['lane']))
    sg = float(lt['sig'][jt]); n_rank = 1 + int(((lt['left'] == side) & (lt['sig'] < sg)).sum())
    same_side = np.flatnonzero(lr['left'] == side); order = [int(lr['ids'][q]) for q in same_side[np.argsort(lr['sig'][same_side])]]
    vsig = {int(lr['ids'][q]): float(lr['sig'][q]) for q in same_side}
    if abs(sg - r['station_m']) > 1e-9 or n_rank != r['n'] or ('left' if side else 'right') != r['side']: issues.append((f'target{t}.station/n/side', (sg, n_rank, side), (r['station_m'], r['n'], r['side'])))
    sim = sims[i]; ideal = vsig[p] - D; evaluable = bool(dstation(lr, sim['Pg'][0]) + 1.0 <= ideal <= dstation(lr, sim['Pg'][-1]))
    if evaluable != r['stop_evaluable']: issues.append((f'target{t}.stop_evaluable', evaluable, r['stop_evaluable']))
    if not evaluable: continue
    checked += 1
    for v in list(VARS) + ['O']:
        rec = r['var'].get(v, {})
        if sim['var'][v] is None:
            if 'outcome' in rec: issues.append((f'target{t}.{v}', 'no simulation', rec['outcome']))
            continue
        counts[v]['n'] += 1; sp = stop_point(*sim['var'][v], sg - D)
        if sp is None:
            out = 'other'
            if not rec.get('no_stop'): issues.append((f'target{t}.{v}.stop', None, rec.get('stop_k')))
        else:
            j, f = sp; pos = sim['Pg'][j] if (f >= 1.0 or j == 0) else (1 - f) * sim['Pg'][j - 1] + f * sim['Pg'][j]
            aim = dstation(lr, pos) + D; arr = min(order, key=lambda b: abs(vsig[b] - aim))
            out = 'right' if arr == p else ('neighbour' if abs(order.index(arr) - order.index(p)) == 1 else 'other')
            if rec.get('no_stop') or rec.get('stop_k') != lr['k0'] + j or abs(rec.get('stop_frac', -9) - f) > 1e-6 or rec.get('arrival') != arr:
                issues.append((f'target{t}.{v}.stop', (lr['k0'] + j, round(f, 6), arr), (rec.get('stop_k'), rec.get('stop_frac'), rec.get('arrival'))))
        counts[v][out] += 1
        if out != rec.get('outcome'): issues.append((f'target{t}.{v}.outcome', out, rec.get('outcome')))
for v, c in counts.items():
    w = SUM['variants'][v]
    if (c['n'], c['right'], c['neighbour'], c['other']) != (w['n'], w['right'], w['neighbour'], w['other']): issues.append((f'{v}.counts', c, {k: w[k] for k in ('n', 'right', 'neighbour', 'other')}))
Path(A.out).mkdir(parents=True, exist_ok=True)
json.dump(dict(tag=TAG, standoff_d_m=D, spacing_m=S, stop_evaluable_checked=checked, counts=counts, issues=[list(map(str, x)) for x in issues]), open(Path(A.out) / f'resim_{TAG}.json', 'w'), indent=1)
for v, c in counts.items(): print(f"  {v:3s} right {c['right']}/{c['n']} neighbour {c['neighbour']} other {c['other']}")
print(f'[resim] {TAG}: {checked} stop-evaluable targets re-simulated; ' + ('ALL AGREE with the curves and records' if not issues else f'{len(issues)} DISAGREEMENTS'))
for x in issues[:40]: print('   ', x)
