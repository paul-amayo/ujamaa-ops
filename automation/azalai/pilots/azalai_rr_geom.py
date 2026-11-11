#!/usr/bin/env python3
"""azalai_rr_geom.py (11 Nov 2026, A100 session) — RSS Azalai pilot "row to relevancy without a keyframe": the GEOMETRY-ONLY step for a
(registry pass, revisit pass) pair, as RR_addendum2_short_gap.md (ad1f663) asks for the short-gap rungs S1 / S2 (registry = the April
11:39 census, no twin, no render): the lane passes, inner rows and lane frames (primary / sequence / oracle [/ sitewide]) with the
functions of azalai_rr_prep.py copied verbatim, the per-lane datum table, the standoff d of the registry pass by its rule, and step 5
of Part 2 of RR_design.md (5c6e5a4): the registry bag carried into the revisit by the datum's transfer; the nearest revisit census bag
in the lane's ground plane is the partner (yes / no); the distance to the partner (the scorer's own arithmetic, azalai_rr_score.py).
Truth: the pilot's record (--pilot) or mutual nearest neighbours within 0.5 m in the December ground plane after both passes'
site-wide fits (placed_<pass>.npz). A pair is evaluable if its registry bag stands in an inner row of a lane of the registry pass and
that lane has lane frames in both passes (there is no twin, so no identity criterion); the others are counted by reason.
The 0.80 line of RR_design.md is applied to the step-5 partner rate at the primary datum (A100 implementation note, committed before
any of these outcomes was computed). No GPU. nerf_new python3.10:
  azalai_rr_geom.py --twin apr1130 --rev apr1400 --out <dir> --suffix _apr1130_apr1400 [--pilot h3_arrival_apr1130.json] [--sitewide]"""
import argparse, json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from scipy.stats import binomtest
ap = argparse.ArgumentParser(); ap.add_argument('--twin', default='apr1130'); ap.add_argument('--rev', default='apr1400'); ap.add_argument('--out', required=True); ap.add_argument('--suffix', default='')
ap.add_argument('--pilot', default=''); ap.add_argument('--sitewide', action='store_true'); a = ap.parse_args()
KL = Path('/home/paperspace/data/klapmuts'); X = KL / 'dec_2025_ten_rows/experimental'; SK = X / 'sankofa/jirs_20261023'; STPG = X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'
Z = np.array([0.0, 0.0, 1.0])
AXIS_Z, JUMP, MINLEN, INNER, SEQ_R, PAIR_R = 0.8, 0.6, 30, 1.5, 0.45, 0.5                              # constants of the design (as azalai_rr_prep.py)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); E = json.load(open(STPG))['experiences']
def cdir(name): return SK / ('camera_depth_feb_fixedreader' if name.startswith('feb') else 'lidar_fixedreader')
def G_of(name): return np.eye(4) if name == 'dec' else np.load(cdir(name) / f'placed_{name}.npz')['G_global'].astype(float)   # pass graph -> December graph (site-wide fit)
def resid_of(name): return 0.0 if name == 'dec' else float(np.load(cdir(name) / f'placed_{name}.npz')['global_fit_resid_med'])
def tf(T, p): return (T[:3, :3] @ np.asarray(p, float).T).T + T[:3, 3]
# ---------------------------------------------------------------- copied verbatim from azalai_rr_prep.py (lane passes, inner rows, lane frames)
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
G = np.linalg.inv(G_of(a.twin)) @ G_of(a.rev); Gi = np.linalg.inv(G)                               # revisit graph -> registry graph by the site-wide fits: truth and oracle only
print(f"[geom] registry {a.twin}: {len(TW['L'])} lane passes, senses {[l['sense'] for l in TW['L']]}; revisit {a.rev}: {len(RV['L'])}, senses {[l['sense'] for l in RV['L']]}", flush=True)
same_lanes = len(TW['L']) == len(RV['L']) and all(x['sense'] == y['sense'] for x, y in zip(TW['L'], RV['L']))
assert same_lanes, 'the lane count or the senses differ between the passes: stop and report (design, Definitions)'
NL = len(TW['L'])
def seq_shift(st, lt, sr, lr):
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
SPACING = float(np.median(gaps))                                                                    # median inner-row bag spacing of the registry pass
DAT = ['primary', 'sequence', 'oracle'] + (['sitewide'] if a.sitewide else []); lanes = []
for i in range(NL):
    t, r = TW['L'][i], RV['L'][i]; e = dict(lane=i + 1, sense=t['sense'], twin_k=[t['k0'], t['k1']], rev_k=[r['k0'], r['k1']], twin_track_rms_m=t['track_rms_m'], rev_track_rms_m=r['track_rms_m'], twin_seen=t['seen'], rev_seen=r['seen'])
    for nm_, x in (('twin', t), ('rev', r)):
        if x['fr'] is not None: e[f'{nm_}_inner'] = dict(left=int(x['fr']['left'].sum()), right=int((~x['fr']['left']).sum()), plane_rms_m=x['fr']['plane_rms_m'], row_sep_m=x['fr']['row_sep_m'], line_rms_m=x['fr']['line_rms_m'],
                                                             length_m=float(np.ptp(x['fr']['station'])), spacing_m=float(np.median(np.concatenate([np.diff(np.sort(x['fr']['station'][x['fr']['left'] == sd])) for sd in (True, False)]))), first_bag=int(x['inner'][x['fr']['first_bag']]))
    if t['F'] is None or r['F'] is None: e['skip'] = 'an inner row is missing on one side: no lane frame'; lanes.append(e); t['T'] = None; continue
    Ft, Fr = t['F'], r['F']; d_seq, n_it, n_m = seq_shift(t['fr']['station'], t['fr']['left'], r['fr']['station'], r['fr']['left'])
    s_or = float(Fr[:3, 2] @ (tf(Gi, Ft[:3, 3]) - Fr[:3, 3]))                                    # station (revisit primary frame) of the registry's row end carried by the site-wide fit
    r['Fd'] = dict(primary=Fr, sequence=shifted_origin(Fr, d_seq), oracle=shifted_origin(Fr, -s_or)); t['T'] = {k: Ft @ np.linalg.inv(v) for k, v in r['Fd'].items()}
    if a.sitewide: t['T']['sitewide'] = G
    cam = Cr[r['k0']:r['k1'] + 1]; mid = cam[len(cam) // 2]; e.update(row_end_minus_oracle_m=dict(primary=s_or, sequence=s_or + d_seq, oracle=0.0), sequence=dict(shift_m=d_seq, iterations=n_it, matches=n_m),
                                                                      datum_failure=bool(abs(s_or) > 0.5 * SPACING), sequence_off_by_more_than_half_spacing=bool(abs(s_or + d_seq) > 0.5 * SPACING), F_twin=Ft.tolist(), F_rev={k: v.tolist() for k, v in r['Fd'].items()}, vs_sitewide={})
    for k, T in t['T'].items():
        dd = (tf(T, cam[:, :3, 3]) - tf(G, cam[:, :3, 3])) @ Ft[:3, [2, 0, 1]]; dm = (tf(T, mid[:3, 3]) - tf(G, mid[:3, 3])) @ Ft[:3, [2, 0, 1]]
        e['vs_sitewide'][k] = dict(middle=dict(along_m=float(dm[0]), lateral_m=float(dm[1]), vertical_m=float(dm[2])), rotation_deg=rotdeg(T[:3, :3], G[:3, :3]),
                                   cameras=dict(along_med=float(np.median(dd[:, 0])), along_min=float(dd[:, 0].min()), along_max=float(dd[:, 0].max()), lateral_med=float(np.median(dd[:, 1])), lateral_absmax=float(np.abs(dd[:, 1]).max()),
                                                vertical_med=float(np.median(dd[:, 2])), vertical_absmax=float(np.abs(dd[:, 2]).max())))
    lanes.append(e); v = e['vs_sitewide']['primary']
    print(f"[geom] lane {i + 1} ({t['sense']:+d}): inner bags registry {e['twin_inner']['left']}+{e['twin_inner']['right']} / revisit {e['rev_inner']['left']}+{e['rev_inner']['right']}; oracle row end in the primary frame {s_or:+.2f} m, sequence {s_or + d_seq:+.2f} m "
          f"({n_m} matches, {n_it} it){' DATUM FAILURE' if e['datum_failure'] else ''}; primary vs site-wide at the middle: along {v['middle']['along_m']:+.2f} lateral {v['middle']['lateral_m']:+.2f} vertical {v['middle']['vertical_m']:+.2f} m, {v['rotation_deg']:.2f} deg", flush=True)
def assign(S):
    o = {}
    for i, l in enumerate(S['L']):
        for b, la in zip(l['inner'], l['lat']):
            if int(b) not in o or abs(la) < o[int(b)][1]: o[int(b)] = (i, abs(float(la)))
    return {b: v[0] for b, v in o.items()}
lane_t = assign(TW); two = sum(sum(int(b) in set(map(int, l['inner'])) for l in TW['L']) > 1 for b in lane_t)
def station(F, p): return float(F[:3, 2] @ (np.asarray(p, float) - F[:3, 3]))
dl = [abs(station(TW['L'][i]['F'], Bt[b]) - station(TW['L'][i]['F'], Ct[max(TW['obs'][b], key=lambda x: x[1])[0], :3, 3])) for b, i in lane_t.items() if TW['L'][i]['F'] is not None and np.isfinite(Ct[max(TW['obs'][b], key=lambda x: x[1])[0]]).all()]
D_ST = float(np.median(dl)); print(f'[geom] standoff d = {D_ST:.3f} m (median over {len(dl)} registry inner-row bags; quartiles {np.percentile(dl, 25):.2f} / {np.percentile(dl, 75):.2f}); median inner-row bag spacing {SPACING:.3f} m; {two} bags stand in the inner rows of two lanes', flush=True)
# ---------------------------------------------------------------- truth pairs (as azalai_rr_prep.py)
okt = np.flatnonzero(TW['fin'])
St = tf(G_of(a.twin), Bt); Sr = tf(G_of(a.rev), Br); okr = np.flatnonzero(np.isfinite(Sr).all(1)); td = cKDTree(St[okt][:, [0, 2]]); tr_ = cKDTree(Sr[okr][:, [0, 2]])
dA, jA = tr_.query(St[okt][:, [0, 2]]); _, jB = td.query(Sr[okr][:, [0, 2]]); mnn = {int(okt[i]): int(okr[jA[i]]) for i in range(len(okt)) if dA[i] <= PAIR_R and jB[jA[i]] == i}
if a.pilot:
    pairs = [(r['dec'], r['apr']) for r in json.load(open(a.pilot))['targets']]; assert all(mnn.get(t) == p for t, p in pairs), "the pilot's pairs are not mutual nearest neighbours here"
else: pairs = sorted(mnn.items())
print(f"[geom] truth: {len(pairs)} pairs ({'the pilot record' if a.pilot else 'mutual nearest neighbours'}; {len(mnn)} mutual nearest neighbours within {PAIR_R} m); site-wide fit residual median registry {resid_of(a.twin):.4f} m, revisit {resid_of(a.rev):.4f} m", flush=True)
# ---------------------------------------------------------------- step 5 of Part 2 (the scorer's arithmetic, azalai_rr_score.py)
finr = np.flatnonzero(np.isfinite(Br).all(1)); lines = []
for t, p in pairs:
    i = lane_t.get(t); r = dict(part='geometry', twin_bag=t, partner=p, lane=None if i is None else i + 1)
    if i is None: r['skip'] = 'registry bag does not stand in an inner row of a lane'
    elif TW['L'][i]['T'] is None: r['skip'] = 'the lane has no lane frame in one of the passes'
    if 'skip' in r: lines.append(r); continue
    L = lanes[i]; Ft = np.array(L['F_twin']); Fp = np.array(L['F_rev']['primary']); r.update(sense=TW['L'][i]['sense'], station_m=station(Ft, Bt[t]), geometry={})
    Q = (Br[finr] - Fp[:3, 3]) @ Fp[:3, [0, 2]]
    for dn in DAT:
        Ti = np.linalg.inv(G) if dn == 'sitewide' else np.array(L['F_rev'][dn]) @ np.linalg.inv(Ft); q = (Ti[:3, :3] @ Bt[t] + Ti[:3, 3] - Fp[:3, 3]) @ Fp[:3, [0, 2]]
        dd = np.linalg.norm(Q - q, axis=1); j = int(np.argmin(dd)); r['geometry'][dn] = dict(nearest=int(finr[j]), nearest_dist_m=float(dd[j]), partner_dist_m=float(np.linalg.norm((Br[p] - Fp[:3, 3]) @ Fp[:3, [0, 2]] - q)), is_partner=bool(int(finr[j]) == p))
    lines.append(r)
with open(OUT / f'rr_geom_targets{a.suffix}.jsonl', 'w') as f:
    for r in lines: f.write(json.dumps(r) + '\n')
ev = [r for r in lines if 'skip' not in r]; fail = {l['lane'] for l in lanes if l.get('datum_failure')}
def ci(k, n): c = binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact'); return [float(c.low), float(c.high)]
def rate(R, dn):
    k = sum(r['geometry'][dn]['is_partner'] for r in R); n = len(R)
    return dict(n=n, partner=k, rate=k / n, ci95_exact=ci(k, n), lower_bound_above_0_80=bool(ci(k, n)[0] > 0.80), partner_dist_median_m=float(np.median([r['geometry'][dn]['partner_dist_m'] for r in R])),
                partner_dist_p90_m=float(np.percentile([r['geometry'][dn]['partner_dist_m'] for r in R], 90))) if n else dict(n=0)
sub = dict(all=ev, without_datum_failure_lanes=[r for r in ev if r['lane'] not in fail], datum_failure_lanes=[r for r in ev if r['lane'] in fail])
S = dict(registry=a.twin, revisit=a.rev, design='RR_design.md 5c6e5a4 (step 5 of Part 2) + RR_addendum_ladder.md 8fbc128 + RR_addendum2_short_gap.md ad1f663', datums=DAT, n_lanes=NL, standoff_d_m=D_ST, standoff_n=len(dl),
         spacing_m=SPACING, half_spacing_m=0.5 * SPACING, bags_in_two_lanes=two, sitewide_fit_resid_med_m=dict(registry=resid_of(a.twin), revisit=resid_of(a.rev)), truth_readable=bool(max(resid_of(a.twin), resid_of(a.rev)) <= 0.15),
         n_pairs=len(pairs), mutual_nn=len(mnn), evaluable=len(ev), not_evaluable=dict(Counter(r['skip'] for r in lines if 'skip' in r)), evaluable_by_lane=dict(sorted(Counter(r['lane'] for r in ev).items())),
         datum_failures=sorted(fail), sequence_off_lanes=[l['lane'] for l in lanes if l.get('sequence_off_by_more_than_half_spacing')],
         geometry_only={dn: {name: rate(R, dn) for name, R in sub.items()} for dn in DAT},
         per_lane={str(l['lane']): dict(n=sum(r['lane'] == l['lane'] for r in ev), datum_failure=l.get('datum_failure'), row_end_minus_oracle_m=l.get('row_end_minus_oracle_m'),
                                        partner={dn: sum(r['geometry'][dn]['is_partner'] for r in ev if r['lane'] == l['lane']) for dn in DAT}) for l in lanes},
         lanes=[{k: v for k, v in l.items() if k not in ('F_twin', 'F_rev')} for l in lanes])
json.dump(dict(summary=S, lanes_full=lanes, twin_passes=[[l['k0'], l['k1'], l['sense']] for l in TW['L']], rev_passes=[[l['k0'], l['k1'], l['sense']] for l in RV['L']],   # output only (for the plan / row-end figures)
               twin_inner={str(i + 1): dict(bags=[int(b) for b in l['inner']], left=[bool(x) for x in l['fr']['left']] if l['fr'] else None) for i, l in enumerate(TW['L'])},
               rev_inner={str(i + 1): dict(bags=[int(b) for b in l['inner']], left=[bool(x) for x in l['fr']['left']] if l['fr'] else None) for i, l in enumerate(RV['L'])}), open(OUT / f'rr_geom{a.suffix}.json', 'w'), indent=1)
print(f"== {a.twin} -> {a.rev}: {len(ev)} of {len(pairs)} evaluable; not evaluable {S['not_evaluable']}; truth readable {S['truth_readable']}; datum failure lanes {S['datum_failures']}; sequence off lanes {S['sequence_off_lanes']}")
for dn in DAT: print(f"   step 5, {dn:9s}: " + '; '.join(f"{name} {v.get('partner')}/{v['n']}" + (f" = {v['rate']:.3f} [{v['ci95_exact'][0]:.3f}, {v['ci95_exact'][1]:.3f}]" if v['n'] else '') for name, v in S['geometry_only'][dn].items()))
