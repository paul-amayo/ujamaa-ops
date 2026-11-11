#!/usr/bin/env python3
"""azalai_rr_prep.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe", step 1: GEOMETRY ONLY, no outcome is
computed here. Implements the definitions of Azalai_RSS/numbers/pilots_20261111/RR_design.md (5c6e5a4) and its addendum
RR_addendum_ladder.md (8fbc128) for one (twin, revisit pass) pair and writes what the renderer and the scorer need:
  * lane passes of both seasons (maximal runs of consecutive keyframes with |z of the optical axis| > 0.8 in one sense, no sideways
    jump over 0.6 m, at least 30 long); lane i of the revisit is lane i of the twin (count and senses are checked);
  * inner rows (census bags seen from the pass, within 1.5 m of the line through its camera centres) and the LANE FRAME of each
    season and lane from those bags only: plane (normal n up), row axis a (two parallel lines, one slope, to the site's +z),
    lateral origin midway, l = n x a, along-row origin = the first inner-row bag met in the revisit's driving sense (PRIMARY);
    the SEQUENCE datum (primary refined by 1-D matching of the two seasons' inner-row stations, same side, mutual nearest within
    0.45 m, median shift, until under 1 mm, at most 20 times); the ORACLE datum (the twin's row end carried into the revisit by the
    site-wide fit; every other degree of freedom still from the lane frame). With --sitewide a fourth comparator: the stop frame
    carried by the site-wide fit itself, all six degrees of freedom (the pose transfer of the H3 pilot);
  * the standoff d (median along-row distance between a twin inner-row bag and its best-view keyframe, twin registry alone);
  * graph -> twin cell exactly as azalai_h3_twin_prep.py (one rigid map + the residual of the nearest keyframe of the lane's cell
    with the same heading), the twin identity of each census bag (mutual nearest anchor within 0.5 m), the naming-vote check;
  * Part 2: for every truth pair the lane, the reasons it cannot be evaluated, and per datum the stop frame (the revisit keyframe
    of the lane whose station is nearest to the target's station minus d in the driving sense) and the believed poses of the
    sweep (the stop frame's pose through F_twin F_rev^-1, displaced along the row by e, + = the driving sense);
  * Part 1 (--part1, twin only): for every twin anchor in the vocabulary of a cell and within 1.5 m of that cell's lane line the
    free poses (centre line, 0.3 m left / right of the driving sense; heading along the row; height, roll, pitch of the nearest
    training keyframe) and the two comparators (nearest held-out / training keyframe of the cell to that station);
  * the render jobs. Prints geometry diagnostics (frames against the oracle per lane, d, counts by reason) and nothing else.
Truth pairs: the pilot's record (--pilot, the 552 of the pair of record) or mutual nearest neighbours within 0.5 m in the December
ground plane after the site-wide fits (placed_<pass>.npz). nerf_new python3.10:
  azalai_rr_prep.py --twin dec --rev apr1130 --out <dir> [--pilot h3_arrival_apr1130.json] [--registry registry_v0.json]
                    [--h3prep prep.json] [--part1] [--sitewide]"""
import argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--twin', default='dec'); ap.add_argument('--rev', default='apr1130'); ap.add_argument('--out', required=True); ap.add_argument('--pilot', default='')
ap.add_argument('--registry', default=''); ap.add_argument('--h3prep', default=''); ap.add_argument('--part1', action='store_true'); ap.add_argument('--sitewide', action='store_true'); a = ap.parse_args()
KL = Path('/home/paperspace/data/klapmuts'); X = KL / 'dec_2025_ten_rows/experimental'; SK = X / 'sankofa/jirs_20261023'; STPG = X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'
INV = '/home/paperspace/logs/identity_inventory_v2.jsonl'; GL = np.diag([1.0, -1.0, -1.0, 1.0]); Z = np.array([0.0, 0.0, 1.0])
ES = [0.0, 0.15, -0.15, 0.30, -0.30, 0.45, -0.45, 0.90, -0.90]                      # the sweep of the design, metres, + = the driving sense
AXIS_Z, JUMP, MINLEN, INNER, SEQ_R, PAIR_R, NAME_R, SIDE = 0.8, 0.6, 30, 1.5, 0.45, 0.5, 0.5, 0.3   # constants of the design
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); E = json.load(open(STPG))['experiences']
def cdir(name): return SK / ('camera_depth_feb_fixedreader' if name.startswith('feb') else 'lidar_fixedreader')
def G_of(name): return np.eye(4) if name == 'dec' else np.load(cdir(name) / f'placed_{name}.npz')['G_global'].astype(float)   # pass graph -> December graph (site-wide fit)
def resid_of(name): return 0.0 if name == 'dec' else float(np.load(cdir(name) / f'placed_{name}.npz')['global_fit_resid_med'])
def tf(T, p): return (T[:3, :3] @ np.asarray(p, float).T).T + T[:3, 3]
# ---------------------------------------------------------------- lane passes, inner rows, lane frames
def passes(C):
    ok = np.isfinite(C).all((1, 2)); z = C[:, 2, 2]; x = C[:, 0, 3]; runs = []; cur = []
    for k in range(len(C)):
        s = 0 if (not ok[k] or abs(z[k]) <= AXIS_Z) else int(np.sign(z[k]))
        if s != 0 and cur and s == cur[0][1] and k == cur[-1][0] + 1 and abs(x[k] - x[cur[-1][0]]) <= JUMP: cur.append((k, s))
        else:
            if len(cur) >= MINLEN: runs.append(cur)
            cur = [(k, s)] if s != 0 else []
    if len(cur) >= MINLEN: runs.append(cur)
    return [(r[0][0], r[-1][0], r[0][1]) for r in runs]
def lane_frame(Bi, lat, sense):
    """inner-row bag points Bi (n x 3), their signed offset from the camera line (+ = the +x side), driving sense -> (F lane->graph
    with columns l, n, a and the PRIMARY origin, diagnostics) or None when a side has under two bags"""
    left = lat < 0
    if left.sum() < 2 or (~left).sum() < 2: return None
    c = Bi.mean(0); n = np.linalg.svd(Bi - c)[2][2]; n = -n if n[1] > 0 else n               # up is -y in the camera-convention world
    ea = Z - (Z @ n) * n; ea /= np.linalg.norm(ea); el = np.cross(n, ea); u = (Bi - c) @ el; w = (Bi - c) @ ea
    (m, cl, cr), *_ = np.linalg.lstsq(np.c_[w, left, ~left].astype(float), u, rcond=None)          # two parallel lines: one slope, two intercepts
    ax = ea + m * el; ax /= np.linalg.norm(ax); l = np.cross(n, ax); q0 = c + 0.5 * (cl + cr) * el
    s = (Bi - q0) @ ax; s0 = float(s.min() if sense > 0 else s.max()); F = np.eye(4); F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = l, n, ax, q0 + s0 * ax
    return F, dict(station=s - s0, left=left, lateral=(Bi - q0) @ l, plane_rms_m=float(np.std((Bi - c) @ n)), row_sep_m=float(cl - cr),
                   line_rms_m=float(np.std(u - m * w - np.where(left, cl, cr))), first_bag=int(np.argmin(s) if sense > 0 else np.argmax(s)))
def season(name):
    C = np.array(E[name]['c2w'], float); z = np.load(cdir(name) / f'census_{name}.npz'); B = z['bags_xyz'].astype(float); fin = np.isfinite(B).all(1)
    obs = defaultdict(list); frm = defaultdict(set)
    for b, k, n in zip(z['obs_bag'], z['obs_k'], z['obs_n']): obs[int(b)].append((int(k), int(n))); frm[int(k)].add(int(b))
    L = []
    for k0, k1, s in passes(C):
        P = C[k0:k1 + 1, :3, 3]; m = P[:, [0, 2]].mean(0); d = np.linalg.svd(P[:, [0, 2]] - m)[2][0]; d = d * np.sign(d[1]); nrm = np.array([d[1], -d[0]])
        seen = np.array(sorted(b for b in set().union(*[frm[k] for k in range(k0, k1 + 1)]) if fin[b]), int); lat = (B[seen][:, [0, 2]] - m) @ nrm if len(seen) else np.zeros(0)
        inn = np.abs(lat) <= INNER; fr = lane_frame(B[seen[inn]], lat[inn], s) if inn.any() else None
        L.append(dict(k0=k0, k1=k1, sense=s, m=m, nrm=nrm, track_rms_m=float(np.std((P[:, [0, 2]] - m) @ nrm)), seen=len(seen), inner=seen[inn], lat=lat[inn], F=None if fr is None else fr[0], fr=None if fr is None else fr[1]))
    return dict(C=C, B=B, fin=fin, obs=obs, L=L)
TW, RV = season(a.twin), season(a.rev); Ct, Bt, Cr, Br = TW['C'], TW['B'], RV['C'], RV['B']
G = np.linalg.inv(G_of(a.twin)) @ G_of(a.rev); Gi = np.linalg.inv(G)                               # revisit graph -> twin graph by the site-wide fits: truth and oracle only
print(f"[prep] twin {a.twin}: {len(TW['L'])} lane passes, senses {[l['sense'] for l in TW['L']]}; revisit {a.rev}: {len(RV['L'])}, senses {[l['sense'] for l in RV['L']]}", flush=True)
same_lanes = len(TW['L']) == len(RV['L']) and all(x['sense'] == y['sense'] for x, y in zip(TW['L'], RV['L']))
assert same_lanes, 'the lane count or the senses differ between the seasons: stop and report (design, Definitions)'
NL = len(TW['L'])
def seq_shift(st, lt, sr, lr):
    """one-dimensional matching of the two seasons' inner-row stations -> (shift to add to the revisit's stations, iterations, matches)"""
    d = 0.0; it = 0; nm = 0
    for it in range(1, 21):
        df = []
        for sd in (True, False):
            x, y = st[lt == sd], sr[lr == sd] + d
            if not len(x) or not len(y): continue
            i = np.abs(x[:, None] - y[None]).argmin(1); j = np.abs(x[:, None] - y[None]).argmin(0)
            df += [x[p] - y[i[p]] for p in range(len(x)) if j[i[p]] == p and abs(x[p] - y[i[p]]) <= SEQ_R]
        nm = len(df)
        if not df: break
        step = float(np.median(df)); d += step
        if abs(step) < 1e-3: break
    return d, it, nm
def shifted_origin(F, ds): Fn = F.copy(); Fn[:3, 3] = F[:3, 3] - ds * F[:3, 2]; return Fn                # stations grow by ds
def rotdeg(Ra, Rb): return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))
gaps = np.concatenate([np.diff(np.sort(l['fr']['station'][l['fr']['left'] == sd])) for l in TW['L'] if l['fr'] is not None for sd in (True, False)])
SPACING = float(np.median(gaps))                                                                    # median inner-row bag spacing of the twin's registry
DAT = ['primary', 'sequence', 'oracle'] + (['sitewide'] if a.sitewide else []); lanes = []
for i in range(NL):
    t, r = TW['L'][i], RV['L'][i]; e = dict(lane=i + 1, sense=t['sense'], twin_k=[t['k0'], t['k1']], rev_k=[r['k0'], r['k1']], twin_track_rms_m=t['track_rms_m'], rev_track_rms_m=r['track_rms_m'], twin_seen=t['seen'], rev_seen=r['seen'])
    for nm_, x in (('twin', t), ('rev', r)):
        if x['fr'] is not None: e[f'{nm_}_inner'] = dict(left=int(x['fr']['left'].sum()), right=int((~x['fr']['left']).sum()), plane_rms_m=x['fr']['plane_rms_m'], row_sep_m=x['fr']['row_sep_m'], line_rms_m=x['fr']['line_rms_m'],
                                                             length_m=float(np.ptp(x['fr']['station'])), spacing_m=float(np.median(np.concatenate([np.diff(np.sort(x['fr']['station'][x['fr']['left'] == sd])) for sd in (True, False)]))), first_bag=int(x['inner'][x['fr']['first_bag']]))
    if t['F'] is None or r['F'] is None: e['skip'] = 'an inner row is missing on one side: no lane frame'; lanes.append(e); t['T'] = None; continue
    Ft, Fr = t['F'], r['F']; d_seq, n_it, n_m = seq_shift(t['fr']['station'], t['fr']['left'], r['fr']['station'], r['fr']['left'])
    s_or = float(Fr[:3, 2] @ (tf(Gi, Ft[:3, 3]) - Fr[:3, 3]))                                    # station (revisit primary frame) of the twin's row end carried by the site-wide fit
    r['Fd'] = dict(primary=Fr, sequence=shifted_origin(Fr, d_seq), oracle=shifted_origin(Fr, -s_or)); t['T'] = {k: Ft @ np.linalg.inv(v) for k, v in r['Fd'].items()}
    if a.sitewide: t['T']['sitewide'] = G
    cam = Cr[r['k0']:r['k1'] + 1]; mid = cam[len(cam) // 2]; e.update(row_end_minus_oracle_m=dict(primary=s_or, sequence=s_or + d_seq, oracle=0.0), sequence=dict(shift_m=d_seq, iterations=n_it, matches=n_m),
                                                                      datum_failure=bool(abs(s_or) > 0.5 * SPACING), F_twin=Ft.tolist(), F_rev={k: v.tolist() for k, v in r['Fd'].items()}, vs_sitewide={})
    for k, T in t['T'].items():   # the lane-frame transfer against the site-wide fit: at the lane's middle keyframe, and over all the lane's revisit keyframes
        dd = (tf(T, cam[:, :3, 3]) - tf(G, cam[:, :3, 3])) @ Ft[:3, [2, 0, 1]]; dm = (tf(T, mid[:3, 3]) - tf(G, mid[:3, 3])) @ Ft[:3, [2, 0, 1]]
        e['vs_sitewide'][k] = dict(middle=dict(along_m=float(dm[0]), lateral_m=float(dm[1]), vertical_m=float(dm[2])), rotation_deg=rotdeg(T[:3, :3], G[:3, :3]),
                                   cameras=dict(along_med=float(np.median(dd[:, 0])), along_min=float(dd[:, 0].min()), along_max=float(dd[:, 0].max()), lateral_med=float(np.median(dd[:, 1])), lateral_absmax=float(np.abs(dd[:, 1]).max()),
                                                vertical_med=float(np.median(dd[:, 2])), vertical_absmax=float(np.abs(dd[:, 2]).max())))
    lanes.append(e); v = e['vs_sitewide']['primary']
    print(f"[prep] lane {i + 1} ({t['sense']:+d}): inner bags twin {e['twin_inner']['left']}+{e['twin_inner']['right']} / revisit {e['rev_inner']['left']}+{e['rev_inner']['right']}; row end minus oracle: primary {s_or:+.2f} m, sequence {s_or + d_seq:+.2f} m "
          f"({n_m} matches, {n_it} it){' DATUM FAILURE' if e['datum_failure'] else ''}; primary vs site-wide at the middle: along {v['middle']['along_m']:+.2f} lateral {v['middle']['lateral_m']:+.2f} vertical {v['middle']['vertical_m']:+.2f} m, {v['rotation_deg']:.2f} deg; "
          f"oracle-datum lateral {e['vs_sitewide']['oracle']['middle']['lateral_m']:+.2f} vertical {e['vs_sitewide']['oracle']['middle']['vertical_m']:+.2f} along {e['vs_sitewide']['oracle']['cameras']['along_min']:+.2f}..{e['vs_sitewide']['oracle']['cameras']['along_max']:+.2f}", flush=True)
# a bag standing in the inner rows of two lanes goes to the lane whose camera line is nearer (truth-free)
def assign(S):
    o = {}
    for i, l in enumerate(S['L']):
        for b, la in zip(l['inner'], l['lat']):
            if int(b) not in o or abs(la) < o[int(b)][1]: o[int(b)] = (i, abs(float(la)))
    return {b: v[0] for b, v in o.items()}
lane_t = assign(TW); two = sum(sum(int(b) in set(map(int, l['inner'])) for l in TW['L']) > 1 for b in lane_t)
def station(F, p): return float(F[:3, 2] @ (np.asarray(p, float) - F[:3, 3]))
# standoff d: twin registry alone
dl = [abs(station(TW['L'][i]['F'], Bt[b]) - station(TW['L'][i]['F'], Ct[max(TW['obs'][b], key=lambda x: x[1])[0], :3, 3])) for b, i in lane_t.items() if TW['L'][i]['F'] is not None and np.isfinite(Ct[max(TW['obs'][b], key=lambda x: x[1])[0]]).all()]
D_ST = float(np.median(dl)); print(f'[prep] standoff d = {D_ST:.3f} m (median over {len(dl)} twin inner-row bags; quartiles {np.percentile(dl, 25):.2f} / {np.percentile(dl, 75):.2f}); median inner-row bag spacing {SPACING:.3f} m; {two} bags stand in the inner rows of two lanes', flush=True)
# ---------------------------------------------------------------- the twin's cells (December build: as azalai_h3_twin_prep.py)
inv = [json.loads(l) for l in open(INV)]; tsd = np.array(E[a.twin]['ts_ms']); cells = {}
if a.twin == 'dec':
    HJ = X / 'identity/marker_hierarchy.json'
    for c in [c for c in inv if c['survey'] == 'dec_2025_ten_rows']:
        sup = Path(c['sup']); name = sup.parent.parent.name; ws = Path(c['ws']); tj = json.load(open(ws / 'transforms.json')); st = json.load(open(X / name / 'stamps.json'))
        P = {Path(f['file_path']).name: np.array(f['transform_matrix'], float) @ GL for f in tj['frames']}; by_ts = {int(st[n]): n for n in P}
        cells[name] = dict(ws=str(ws), sup=str(sup), config=json.load(open(Path(c['identity']) / 'W.json'))['config'], K=[tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy']], wh=[tj['w'], tj['h']], P=P,
                           kf={k: by_ts[int(t)] for k, t in enumerate(tsd) if int(t) in by_ts and np.isfinite(Ct[k]).all()}, test=set(json.load(open(ws / 'split.json'))['test']), idd=dict(canon=Path(c['identity']), bag=ws / 'identity_masks_only_v1'))
else: raise SystemExit(f'twin {a.twin}: only the December build is implemented. The April 10:54 twin (apr_2026_zed) is four overlapping spatial chunks plus one lane chunk, one site-wide vocabulary, a hierarchy in the prod pose frame and no bag-level identity; it is not brought under the rule here')
for name, c in cells.items():
    c['bank'] = {}
    for tw, idd in c['idd'].items():
        if not (idd / 'text_bank.npz').exists(): continue
        b = np.load(idd / 'text_bank.npz', allow_pickle=True); words = set(b['words']); wt = json.loads(str(b['meta']))['word_table']
        c['bank'][tw] = dict(identity=str(idd), ids=sorted(int(k) for k, w in wt.items() if int(k) < 10000 and w in words))
    ov = [sum(l['k0'] <= k <= l['k1'] for k in c['kf']) for l in TW['L']]; c['lane'] = int(np.argmax(ov)); c['maps'] = sorted(p.name for p in Path(c['sup']).glob('f_*.png')) if c.get('sup') else []
    print(f"[prep] cell {name}: lane {c['lane'] + 1} ({max(ov)} of its {len(c['kf'])} graph keyframes in the pass), {len(c['P'])} frames, {len(c['test'])} held-out; plant addresses " + ', '.join(f"{tw} {len(v['ids'])}" for tw, v in c['bank'].items()), flush=True)
cell_of = {c['lane']: n for n, c in cells.items()}; assert len(cell_of) == len(cells), 'two cells claim one lane'
kf_cell = {}
for n, c in cells.items():
    for k in c['kf']: kf_cell.setdefault(k, n)
KF = np.array(sorted(kf_cell)); a_, b_ = Ct[KF][:, :3, 3], np.array([cells[kf_cell[int(k)]]['P'][cells[kf_cell[int(k)]]['kf'][int(k)]][:3, 3] for k in KF])
ma, mb = a_.mean(0), b_.mean(0); U, _, Vt = np.linalg.svd((a_ - ma).T @ (b_ - mb)); TR = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T; Tt = mb - TR @ ma
res = np.linalg.norm((TR @ a_.T).T + Tt - b_, axis=1); print(f'[prep] graph -> cell world: {len(KF)} keyframes, residual median {np.median(res):.3f} m, p90 {np.percentile(res, 90):.3f}, max {res.max():.3f}', flush=True)
for n, c in cells.items():
    c['ks'] = np.array(sorted(c['kf'])); c['kp'] = Ct[c['ks']][:, :3, 3]; c['kz'] = Ct[c['ks']][:, :3, 2]; c['Pk'] = {int(k): c['P'][c['kf'][int(k)]] for k in c['ks']}
    c['dk'] = {k: p[:3, 3] - (TR @ Ct[k][:3, 3] + Tt) for k, p in c['Pk'].items()}; c['Ck'] = {k: (TR @ Ct[k][:3, :3]).T @ p[:3, :3] for k, p in c['Pk'].items()}
    c['kc'] = np.array([c['Pk'][int(k)][:3, 3] for k in c['ks']]); c['train'] = np.array([p[:3, 3] for n_, p in c['P'].items() if n_ not in c['test']]); c['train_z'] = np.array([p[:3, 2] for n_, p in c['P'].items() if n_ not in c['test']])
def anchor(pose, c):
    """nearest keyframe of the cell with the same heading (optical axes within 90 deg) -> (k, distance m)"""
    d = np.linalg.norm(c['kp'] - pose[:3, 3], axis=1); d[(c['kz'] @ pose[:3, 2]) <= 0] = np.inf; i = int(np.argmin(d)); return (int(c['ks'][i]), float(d[i])) if np.isfinite(d[i]) else None
def to_cell(pose, c, at=None):
    k, dist = anchor(pose, c) if at is None else (at, 0.0); q = np.eye(4); q[:3, :3] = TR @ pose[:3, :3] @ c['Ck'][k]; q[:3, 3] = TR @ pose[:3, 3] + Tt + c['dk'][k]; return q, k, dist
chk = max(np.abs(to_cell(Ct[int(k)], c, int(k))[0] - c['Pk'][int(k)]).max() for c in cells.values() for k in c['ks']); orth = max(np.abs(Ct[int(k)][:3, :3] @ Ct[int(k)][:3, :3].T - np.eye(3)).max() for k in KF)
print(f'[prep] graph -> cell at the keyframes themselves: max difference {chk:.1e} (graph rotations orthonormal to {orth:.1e})', flush=True); assert chk < 1e-3
def near_train(q, c): i = int(np.argmin(np.linalg.norm(c['train'] - q[:3, 3], axis=1))); return float(np.linalg.norm(c['train'][i] - q[:3, 3])), float(np.degrees(np.arccos(np.clip(c['train_z'][i] @ q[:3, 2], -1, 1))))
# ---------------------------------------------------------------- twin identity of each twin census bag (mutual nearest anchor), naming vote
Hj = json.load(open(HJ)); O = np.array([o['xyz'] for o in Hj['objects']], float); oid = [int(o['id']) for o in Hj['objects']]; opos = {i: O[j] for j, i in enumerate(oid)}
okt = np.flatnonzero(TW['fin']); Bc = (TR @ Bt[okt].T).T + Tt; dO, jO = cKDTree(O[:, :2]).query(Bc[:, :2]); _, jB_ = cKDTree(Bc[:, :2]).query(O[:, :2])
twin_id = {int(okt[i]): oid[jO[i]] for i in range(len(okt)) if dO[i] <= NAME_R and jB_[jO[i]] == i}
print(f'[prep] twin identity: {len(twin_id)} of {len(okt)} twin census bags are the mutual nearest neighbour of a twin anchor within {NAME_R} m; {len(O)} anchors', flush=True)
votes = defaultdict(Counter)
for name, c in cells.items():
    fx, fy, cx, cy = c['K']
    for n in c['maps'][::3]:
        if n not in c['P']: continue
        w2c = np.linalg.inv(c['P'][n]); q = (w2c[:3, :3] @ Bc.T).T + w2c[:3, 3]; m = (q[:, 2] > 0.5) & (q[:, 2] < 3.0)
        if not m.any(): continue
        u = fx * q[:, 0] / q[:, 2] + cx; v = fy * q[:, 1] / q[:, 2] + cy; m &= (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        if not m.any(): continue
        ids = np.array(Image.open(Path(c['sup']) / n))
        for b, uu, vv in zip(okt[m], u[m].astype(int), v[m].astype(int)):
            if ids[vv, uu] < 10000: votes[int(b)][int(ids[vv, uu])] += 1
vt = {b: c.most_common(1)[0][0] for b, c in votes.items() if sum(c.values()) >= 3}
def vote(t): return 'none' if t not in twin_id else ('no votes' if t not in vt else ('agree' if vt[t] == twin_id[t] else 'disagree'))
print(f'[prep] naming check: of {sum(b in twin_id for b in vt)} named bags with >= 3 supervision votes, the vote majority is the same identity for {sum(vt[b] == twin_id[b] for b in vt if b in twin_id)}', flush=True)
if a.h3prep:   # the H3 pilot's prep on the same twin: same map, same names
    H3 = json.load(open(a.h3prep)); assert np.abs(np.array(H3['T_graph_to_cell']['R']) - TR).max() < 1e-9 and {int(k): v for k, v in H3['twin_id'].items()} == twin_id, 'graph -> cell map or twin identity differs from the H3 pilot'
    h3v = {p['dec']: p['naming_vote'] for p in H3['pairs']}; assert all(vote(t) == v for t, v in h3v.items()), 'naming vote differs from the H3 pilot'; print('[prep] graph -> cell map, twin identity and naming votes equal those of the H3 pilot', flush=True)
# ---------------------------------------------------------------- truth pairs
St = tf(G_of(a.twin), Bt); Sr = tf(G_of(a.rev), Br); okr = np.flatnonzero(np.isfinite(Sr).all(1)); td = cKDTree(St[okt][:, [0, 2]]); tr_ = cKDTree(Sr[okr][:, [0, 2]])
dA, jA = tr_.query(St[okt][:, [0, 2]]); _, jB = td.query(Sr[okr][:, [0, 2]]); mnn = {int(okt[i]): int(okr[jA[i]]) for i in range(len(okt)) if dA[i] <= PAIR_R and jB[jA[i]] == i}
if a.pilot:
    pairs = [(r['dec'], r['apr']) for r in json.load(open(a.pilot))['targets']]; assert all(mnn.get(t) == p for t, p in pairs), "the pilot's pairs are not mutual nearest neighbours here"
else: pairs = sorted(mnn.items())
print(f"[prep] truth: {len(pairs)} pairs ({'the pilot record' if a.pilot else 'mutual nearest neighbours'}; {len(mnn)} mutual nearest neighbours within {PAIR_R} m); site-wide fit residual median twin {resid_of(a.twin):.3f} m, revisit {resid_of(a.rev):.3f} m", flush=True)
reg = {b['id']: b for b in json.load(open(a.registry))['bags'] if isinstance(b['id'], int)} if a.registry else {}
# ---------------------------------------------------------------- Part 2: stop frames and believed poses
jobs = []; targets = []
def along(pose, d): q = pose.copy(); q[:3, 3] = q[:3, 3] + d; return q
for t, p in pairs:
    i = lane_t.get(t); ti = twin_id.get(t); e = dict(twin_bag=t, partner=p, twin_id=ti, lane=None if i is None else i + 1, naming_vote=vote(t), growth=(reg.get(t) or {}).get('growth') if reg else None)
    cell = cell_of.get(i) if i is not None else None; e['cell'] = cell; e['in_vocab'] = {tw: bool(ti is not None and cell is not None and ti in cells[cell]['bank'][tw]['ids']) for tw in (cells[cell]['bank'] if cell else {})}
    if ti is None: e['skip'] = 'target has no identity in the twin'
    elif i is None: e['skip'] = 'target does not stand in an inner row of a lane'
    elif cell is None: e['skip'] = "the target's lane has no twin cell"
    elif not e['in_vocab'].get('canon'): e['skip'] = "target's identity is not in the vocabulary of the lane's cell"
    elif TW['L'][i]['T'] is None: e['skip'] = 'the lane has no lane frame in one of the seasons'
    if 'skip' in e: targets.append(e); continue
    Lt, Lr = TW['L'][i], RV['L'][i]; Ft = Lt['F']; sn = Lt['sense']; c = cells[cell]; s_t = station(Ft, Bt[t]); s_stop = s_t - D_ST * sn; ks = np.arange(Lr['k0'], Lr['k1'] + 1)
    e.update(sense=sn, station_m=s_t, lateral_m=float(Ft[:3, 0] @ (Bt[t] - Ft[:3, 3])), stop_station_m=s_stop, datum={})
    for dn in DAT:
        T = Lt['T'][dn]; sk = (tf(T, Cr[ks][:, :3, 3]) - Ft[:3, 3]) @ Ft[:3, 2]                  # stations of the lane's revisit keyframes (the revisit lane frame's, read in the twin's)
        if dn != 'sitewide': sk = (Cr[ks][:, :3, 3] - Lr['Fd'][dn][:3, 3]) @ Lr['Fd'][dn][:3, 2]
        j = int(np.argmin(np.abs(sk - s_stop))); k = int(ks[j]); Pb = T @ Cr[k]; de = dict(stop_k=k, stop_miss_m=float(sk[j] - s_stop), jobs=[], anchor_k=[], anchor_dist_m=[])
        for n_e, ee in enumerate(ES):
            q, ak, ad = to_cell(along(Pb, ee * sn * Ft[:3, 2]), c); name = f'p2_{t}_{dn}_{n_e}'; de['jobs'].append(name); de['anchor_k'].append(ak); de['anchor_dist_m'].append(ad)
            jobs.append(dict(job=name, cell=cell, c2w=q.tolist(), target=ti, full=(ee == 0.0), rgb=(ee == 0.0), depth=False))
        e['datum'][dn] = de
    targets.append(e)
# ---------------------------------------------------------------- Part 1: free poses in the twin
p1 = []
if a.part1:
    kfjobs = set()
    def rot_about(n, ang): Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]]); return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
    for name, c in cells.items():
        i = c['lane']; Lt = TW['L'][i]; Ft = Lt['F']
        if Ft is None: continue
        sn = Lt['sense']; l, n, ax, o = Ft[:3, 0], Ft[:3, 1], Ft[:3, 2], Ft[:3, 3]; inpass = (c['ks'] >= Lt['k0']) & (c['ks'] <= Lt['k1']); held = np.array([c['kf'][int(k)] in c['test'] for k in c['ks']])
        kst = (c['kp'] - o) @ ax; klat = (c['kp'] - o) @ l; bag_of = {v: b for b, v in twin_id.items()}
        for j, an in enumerate(oid):
            if an not in c['bank']['canon']['ids']: continue
            kn = int(np.argmin(np.linalg.norm(c['kc'] - O[j], axis=1))); pg = TR.T @ (O[j] - Tt - c['dk'][int(c['ks'][kn])])     # the anchor in the graph (inverse of the anchored map)
            lat = float((pg[[0, 2]] - Lt['m']) @ Lt['nrm'])
            if abs(lat) > INNER: continue
            s_t = float(ax @ (pg - o)); s_p = s_t - D_ST * sn; b = bag_of.get(an); e = dict(cell=name, lane=i + 1, twin_id=an, anchor=O[j].tolist(), station_m=s_t, lateral_m=float(l @ (pg - o)), line_offset_m=lat,
                                                                                           twin_bag=b, census_inner=bool(b is not None and lane_t.get(b) == i), pose_station_m=s_p, poses={})
            for tag, off in (('C', 0.0), ('L', SIDE * sn), ('R', -SIDE * sn)):
                tr = np.flatnonzero(inpass & ~held); kk = tr[int(np.argmin(np.hypot(kst[tr] - s_p, klat[tr] - off)))]; k = int(c['ks'][kk]); Rk = Ct[k][:3, :3]; hk = Rk[:, 2] - (Rk[:, 2] @ n) * n
                ang = float(np.arctan2(n @ np.cross(hk, sn * ax), hk @ (sn * ax))); pose = np.eye(4); pose[:3, :3] = rot_about(n, ang) @ Rk; pose[:3, 3] = o + s_p * ax + off * l + (n @ (Ct[k][:3, 3] - o)) * n
                q, ak, ad = to_cell(pose, c); dt, hd = near_train(q, c); jn = f'p1_{name}_{an}_{tag}'; e['poses'][tag] = dict(job=jn, c2w=q.tolist(), nearest_train_kf=k, yaw_change_deg=float(np.degrees(ang)), anchor_k=ak, train_dist_m=dt, train_heading_deg=hd)
                jobs.append(dict(job=jn, cell=name, c2w=q.tolist(), target=an, full=True, rgb=(tag == 'C'), depth=True))
            for tag, sel in (('H', inpass & held), ('T', inpass & ~held)):
                cand = np.flatnonzero(sel)
                if not len(cand): continue
                kk = cand[int(np.argmin(np.abs(kst[cand] - s_p)))]; k = int(c['ks'][kk]); fn = c['kf'][k]; q = c['Pk'][k]; dt, hd = near_train(q, c); jn = f"kf_{name}_{Path(fn).stem}"
                e['poses'][tag] = dict(job=jn, c2w=q.tolist(), k=k, frame=fn, station_miss_m=float(kst[kk] - s_p), lateral_m=float(klat[kk]), train_dist_m=dt, train_heading_deg=hd)
                if jn not in kfjobs: kfjobs.add(jn); jobs.append(dict(job=jn, cell=name, c2w=q.tolist(), target=None, full=True, rgb=False, depth=True))
            p1.append(e)
    print(f"[prep] Part 1: {len(p1)} targets (anchor in a cell's vocabulary and within {INNER} m of its lane line; {sum(e['census_inner'] for e in p1)} of them are named census bags of that lane's inner rows), per cell {dict(Counter(e['cell'] for e in p1))}; "
          f"free centre pose to the nearest training camera: median {np.median([e['poses']['C']['train_dist_m'] for e in p1]):.3f} m, {np.median([e['poses']['C']['train_heading_deg'] for e in p1]):.1f} deg; held-out comparator for {sum('H' in e['poses'] for e in p1)}, station miss median {np.median([abs(e['poses']['H']['station_miss_m']) for e in p1 if 'H' in e['poses']]):.2f} m", flush=True)
ev = [e for e in targets if 'skip' not in e]
summ = dict(twin=a.twin, revisit=a.rev, design='RR_design.md 5c6e5a4 + RR_addendum_ladder.md 8fbc128', sweep_m=ES, datums=DAT, n_lanes=NL, standoff_d_m=D_ST, standoff_n=len(dl), spacing_m=SPACING, half_spacing_m=0.5 * SPACING,
            bags_in_two_lanes=two, sitewide_fit_resid_med_m=dict(twin=resid_of(a.twin), revisit=resid_of(a.rev)), truth_readable=bool(max(resid_of(a.twin), resid_of(a.rev)) <= 0.15), n_pairs=len(pairs), mutual_nn=len(mnn),
            evaluable=len(ev), not_evaluable=dict(Counter(e['skip'] for e in targets if 'skip' in e)), evaluable_by_lane=dict(sorted(Counter(e['lane'] for e in ev).items())), naming_vote=dict(Counter(e['naming_vote'] for e in ev)), growth=dict(Counter(str(e['growth']) for e in ev)),
            flags_all_pairs=dict(named=sum(e['twin_id'] is not None for e in targets), inner_row=sum(e['lane'] is not None for e in targets), named_and_inner=sum(e['twin_id'] is not None and e['lane'] is not None for e in targets)),
            datum_failures=[l['lane'] for l in lanes if l.get('datum_failure')], lanes=[{k: v for k, v in l.items() if k not in ('F_twin', 'F_rev')} for l in lanes],
            stop_miss_m={dn: float(np.median([abs(e['datum'][dn]['stop_miss_m']) for e in ev])) for dn in DAT} if ev else {}, anchor_dist_m=float(np.median([e['datum']['primary']['anchor_dist_m'][0] for e in ev])) if ev else None,
            graph_to_cell_resid_m=dict(median=float(np.median(res)), p90=float(np.percentile(res, 90))), twin_named=len(twin_id), part1_targets=len(p1), jobs=len(jobs))
json.dump(dict(summary=summ, twin=a.twin, rev=a.rev, twin_root=E[a.twin]['survey'], rev_root=E[a.rev]['survey'], census_twin=str(cdir(a.twin) / f'census_{a.twin}.npz'), census_rev=str(cdir(a.rev) / f'census_{a.rev}.npz'), stpg=str(STPG),
               G_rev_to_twin=G.tolist(), T_graph_to_cell=dict(R=TR.tolist(), t=Tt.tolist()), hierarchy=str(HJ), lanes=lanes, twin_passes=[[l['k0'], l['k1'], l['sense']] for l in TW['L']], rev_passes=[[l['k0'], l['k1'], l['sense']] for l in RV['L']],
               twin_inner={str(i + 1): dict(bags=[int(b) for b in l['inner']], left=[bool(x) for x in l['fr']['left']] if l['fr'] else None) for i, l in enumerate(TW['L'])},
               rev_inner={str(i + 1): dict(bags=[int(b) for b in l['inner']], left=[bool(x) for x in l['fr']['left']] if l['fr'] else None) for i, l in enumerate(RV['L'])},
               cells={n: dict(ws=c['ws'], sup=c.get('sup'), config=c['config'], K=c['K'], wh=c['wh'], bank=c['bank'], lane=c['lane'] + 1) for n, c in cells.items()},
               twin_ids={str(k): v for k, v in twin_id.items()}, anchors={str(i): opos[i].tolist() for i in oid}, targets=targets, part1=p1, jobs=jobs), open(OUT / 'prep.json', 'w'))
json.dump(summ, open(OUT / 'rr_prep_summary.json', 'w'), indent=1); print(json.dumps({k: v for k, v in summ.items() if k != 'lanes'}, indent=1))
