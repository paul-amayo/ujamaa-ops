#!/usr/bin/env python3
"""azalai_rr_score.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe", step 3: Parts 1 and 2 as fixed in
Azalai_RSS/numbers/pilots_20261111/RR_design.md (5c6e5a4) and RR_addendum_ladder.md (8fbc128) BEFORE this script was first run.
Part 2, for every truth pair of prep.json, every datum (primary / sequence / oracle [/ sitewide]) and every e of the sweep:
  rule R   the twin rendered at the believed pose gives M_t (pixels whose argmax is the target's address); m* = the live SAM 3
           mask of the stop frame with the highest IoU against M_t (IoU > 0; masks at the render scale, every second pixel);
           m* is named by the revisit census: of the census bags whose point projects inside m* at the stop frame (the revisit's
           pose and intrinsics, positive depth, full-resolution mask), the nearest to the camera. Outcome: right plant (the name
           is the target's partner) / wrong plant / unnamed (no census bag inside m*) / no pick (M_t empty, or no live mask
           overlaps it). For the twin of the letter (canon) and the bag-level twin (bag). No threshold.
  geometry only   the target's twin position carried into the revisit by the datum's transfer; the nearest revisit census bag in
           the lane's ground plane is the partner (yes / no); the distance to the partner.
Part 1 (when prep.json carries it), per target and pose (free centre line C, 0.3 m left L / right R, nearest held-out keyframe H,
nearest training keyframe T): lit (M_t not empty), area, and the landing: the pixels of M_t unprojected with the rendered depth,
their median position in the ground plane, the nearest anchor of the twin's hierarchy to it (lands on the target: yes / no), the
landing error, range and bearing from the camera; and the distance between the anchor and the median position of the Gaussians
whose own feature reads its address. Writes rr_targets<suffix>.jsonl (one line per target and part, unevaluable ones included with
their reason) and rr_summary<suffix>.json (outcome counts at every e per datum and twin; subsets; the primary outcome with its exact
95 % interval; sign tests against the oracle; tolerance; per-lane counts; pose-transfer pixel offsets). nerf_new python3.10:
  azalai_rr_score.py <prep.json> <renders dir> <out dir> [--suffix _dec_apr1130]"""
import argparse, sys, json, glob, numpy as np
from pathlib import Path
from collections import Counter
from scipy import ndimage
from scipy.stats import binomtest
import pycocotools.mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths
from rig_calib import load_rig
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('renders'); ap.add_argument('out'); ap.add_argument('--suffix', default=''); a = ap.parse_args()
P = json.load(open(a.prep)); RD = Path(a.renders); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); ES = P['summary']['sweep_m']; DAT = P['summary']['datums']; UNLAB = 65535
E = json.load(open(P['stpg']))['experiences']; Cr = np.array(E[P['rev']]['c2w'], float); Br = np.load(P['census_rev'])['bags_xyz'].astype(float); Bt = np.load(P['census_twin'])['bags_xyz'].astype(float)
finr = np.flatnonzero(np.isfinite(Br).all(1)); KI = load_rig(survey_paths.monos(P['rev_root']))['intrinsics']; G = np.array(P['G_rev_to_twin'], float); LN = {l['lane']: l for l in P['lanes']}
TW = sorted({tw for c in P['cells'].values() for tw in c['bank']}, key=lambda x: x != 'canon')
# ---------------------------------------------------------------- live masks and their census names
ENT = {}
for f in sorted(glob.glob(str(Path(P['rev_root']) / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
    c = json.load(open(f))
    for li, fr in enumerate(c['frames']):
        k = int(fr['kf_idx'])
        if k not in ENT: ENT[k] = [x['rle'] for x in c['frame_entries'].get(str(li), [])]
cache = {}
def live(k):
    """the stop frame's SAM 3 grow-bag masks: full resolution [n, 720, 1280] and at the render scale (every second pixel)"""
    if k not in cache:
        ms = []
        for rle in ENT.get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool))
        if len(cache) > 300: cache.clear()
        M = np.stack(ms) if ms else np.zeros((0, 720, 1280), bool); cache[k] = (M, M[:, ::2, ::2], M[:, ::2, ::2].sum((1, 2)), {})   # A100 fix 20:36 UTC: reshape(0, -1) failed on a stop frame with no mask
    return cache[k]
def name_of(k, i):
    """the revisit census bag m* is named as: of the bags projecting inside mask i at keyframe k, the nearest to the camera"""
    M, _, _, nm = live(k)
    if i not in nm:
        w2c = np.linalg.inv(Cr[k]); q = (w2c[:3, :3] @ Br[finr].T).T + w2c[:3, 3]; ok = q[:, 2] > 0; u = np.full(len(q), -1.0); v = np.full(len(q), -1.0)
        u[ok] = KI['fx'] * q[ok, 0] / q[ok, 2] + KI['cx']; v[ok] = KI['fy'] * q[ok, 1] / q[ok, 2] + KI['cy']; ok &= (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
        ins = np.flatnonzero(ok); ins = ins[M[i][v[ins].astype(int), u[ins].astype(int)]]; rg = np.linalg.norm(q[ins], axis=1)
        nm[i] = (int(finr[ins[int(np.argmin(rg))]]), [int(x) for x in finr[ins]], float(rg.min())) if len(ins) else (None, [], None)
    return nm[i]
def largest(m):
    lab, n = ndimage.label(m, structure=np.ones((3, 3))); return m if n <= 1 else lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
def centroid(m): v, u = np.nonzero(m); return float(u.mean()), float(v.mean())
def rule_r(Mt, k, partner):
    M, Mh, area, _ = live(k); r = dict(mt_px=int(Mt.sum()), n_masks=len(M), iou=0.0, mstar=None, name=None, n_inside=0)
    if r['mt_px'] == 0: r.update(outcome='no pick', reason='M_t empty'); return r
    if not len(M): r.update(outcome='no pick', reason='no live masks'); return r
    inter = (Mh & Mt[None]).reshape(len(M), -1).sum(1); iou = inter / (area + r['mt_px'] - inter); i = int(np.argmax(iou)); r['iou'] = float(iou[i])
    if iou[i] <= 0: r.update(outcome='no pick', reason='no live mask overlaps M_t'); return r
    nm, n_in, rg = name_of(k, i); cu, cv = centroid(Mh[i]); su, sv = centroid(largest(Mt))
    r.update(mstar=i, live_px=int(area[i]), name=nm, n_inside=len(n_in), name_range_m=rg, partner_inside=bool(partner in n_in), off_px=[cu - su, cv - sv])
    r['outcome'] = 'unnamed' if nm is None else ('right plant' if nm == partner else 'wrong plant'); return r
MT = {}
def mt_of(cell):
    if cell not in MT: z = np.load(RD / f'mt_{cell}.npz'); MT[cell] = (dict((str(n), i) for i, n in enumerate(z['jobs'])), {tw: z[tw] for tw in TW if tw in z.files}, tuple(int(x) for x in z['shape']))
    return MT[cell]
def mask(cell, job, tw):
    ix, arr, (h, w) = mt_of(cell); return np.unpackbits(arr[tw][ix[job]], count=h * w).reshape(h, w).astype(bool) if tw in arr else None
lines = []
# ---------------------------------------------------------------- Part 2
for n_t, p in enumerate(P['targets']):
    r = dict(part=2, twin_bag=p['twin_bag'], partner=p['partner'], twin_id=p['twin_id'], lane=p['lane'], cell=p['cell'], naming_vote=p['naming_vote'], growth=p['growth'])
    if 'skip' in p: r['skip'] = p['skip']; lines.append(r); continue
    L = LN[p['lane']]; Ft = np.array(L['F_twin']); Fp = np.array(L['F_rev']['primary']); t = p['twin_bag']; r.update(sense=p['sense'], station_m=p['station_m'], lateral_m=p['lateral_m'], stop_station_m=p['stop_station_m'], geometry={}, R={})
    Q = (Br[finr] - Fp[:3, 3]) @ Fp[:3, [0, 2]]                                                    # revisit census in the lane's ground plane (lateral, along)
    for dn in DAT:
        Ti = np.linalg.inv(G) if dn == 'sitewide' else np.array(L['F_rev'][dn]) @ np.linalg.inv(Ft); q = (Ti[:3, :3] @ Bt[t] + Ti[:3, 3] - Fp[:3, 3]) @ Fp[:3, [0, 2]]
        dd = np.linalg.norm(Q - q, axis=1); j = int(np.argmin(dd)); r['geometry'][dn] = dict(nearest=int(finr[j]), nearest_dist_m=float(dd[j]), partner_dist_m=float(np.linalg.norm((Br[p['partner']] - Fp[:3, 3]) @ Fp[:3, [0, 2]] - q)), is_partner=bool(int(finr[j]) == p['partner']))
        d = p['datum'][dn]; o = dict(stop_k=d['stop_k'], stop_miss_m=d['stop_miss_m'])
        for tw in TW: o[tw] = [dict(e=ES[n_e], **rule_r(mask(p['cell'], job, tw), d['stop_k'], p['partner'])) if p['in_vocab'].get(tw) else dict(e=ES[n_e], mt_px=0, n_masks=len(live(d['stop_k'])[0]), iou=0.0, mstar=None, name=None, n_inside=0, outcome='no pick', reason="target's identity is not in this twin's vocabulary") for n_e, job in enumerate(d['jobs'])]
        r['R'][dn] = o
    lines.append(r)
    if n_t % 100 == 0: print('[rr] part 2', n_t, flush=True)
# ---------------------------------------------------------------- Part 1
if P['part1']:
    O_id = [int(k) for k in P['anchors']]; O = np.array([P['anchors'][str(k)] for k in O_id], float); up = int(np.argmin(O.std(0))); gp = [i for i in range(3) if i != up]   # the cell world's ground plane (the axis the anchors vary least along is up)
    GZ = {c: json.load(open(RD / f'gauss_{c}.json')) for c in P['cells']}; kfc = {}
    def landing(z, tw, tid, c2w, K):
        if tw not in z.files: return None
        Mt = z[tw] == tid; o = dict(lit=bool(Mt.any()), area_px=int(Mt.sum()))
        if not o['lit']: return o
        v, u = np.nonzero(Mt); dep = z['depth'][v, u].astype(float); fx, fy, cx, cy = [x * 0.5 for x in K]; pc = np.c_[(u + 0.5 - cx) / fx * dep, (v + 0.5 - cy) / fy * dep, dep]; pw = (c2w[:3, :3] @ pc.T).T + c2w[:3, 3]
        m = np.median(pw, axis=0); j = int(np.argmin(np.linalg.norm(O[:, gp] - m[gp], axis=1))); tj = O_id.index(tid); dv = m[gp] - c2w[:3, 3][gp]; ax = c2w[:3, 2][gp]
        o.update(land=[float(x) for x in m], nearest_anchor=O_id[j], lands=bool(O_id[j] == tid), err_m=float(np.linalg.norm(O[tj, gp] - m[gp])), range_m=float(np.linalg.norm(dv)),
                 bearing_deg=float(np.degrees(np.arctan2(ax[0] * dv[1] - ax[1] * dv[0], ax @ dv))), depth_med_m=float(np.median(dep))); return o
    for n_t, p in enumerate(P['part1']):
        r = dict(part=1, cell=p['cell'], lane=p['lane'], twin_id=p['twin_id'], twin_bag=p['twin_bag'], census_inner=p['census_inner'], station_m=p['station_m'], lateral_m=p['lateral_m'], poses={}, gauss={}); K = P['cells'][p['cell']]['K']; tj = O_id.index(p['twin_id'])
        for tag, q in p['poses'].items():
            if q['job'] not in kfc:
                if len(kfc) > 60: kfc.clear()
                kfc[q['job']] = np.load(RD / f"{q['job']}.npz")
            z = kfc[q['job']]; o = {k: q[k] for k in ('train_dist_m', 'train_heading_deg', 'station_miss_m', 'k', 'frame', 'nearest_train_kf') if k in q}
            for tw in TW: o[tw] = landing(z, tw, p['twin_id'], np.array(q['c2w'], float), K)
            r['poses'][tag] = o
        for tw in TW:
            gg = GZ[p['cell']].get(tw, {}).get(str(p['twin_id']))
            if gg is not None: r['gauss'][tw] = dict(n=gg['n'], dist_ground_m=None if gg['n'] == 0 else float(np.linalg.norm(np.array(gg['median_xyz'])[gp] - O[tj, gp])), dist_3d_m=None if gg['n'] == 0 else float(np.linalg.norm(np.array(gg['median_xyz']) - O[tj])))
        lines.append(r)
        if n_t % 200 == 0: print('[rr] part 1', n_t, flush=True)
with open(OUT / f'rr_targets{a.suffix}.jsonl', 'w') as f:
    for r in lines: f.write(json.dumps(r) + '\n')
# ---------------------------------------------------------------- summary
OC = ('right plant', 'wrong plant', 'unnamed', 'no pick')
def ci(k, n): c = binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact'); return [float(c.low), float(c.high)]
def sign(g, b): return float(binomtest(g, g + b).pvalue) if g + b else None
def counts(R, dn, tw):
    o = {}
    for n_e, e in enumerate(ES):
        c = Counter(r['R'][dn][tw][n_e]['outcome'] for r in R); o[f'{e:+.2f}'] = dict(n=len(R), **{k: c.get(k, 0) for k in OC}, rate=(c.get('right plant', 0) / len(R)) if R else None)
    return o
def tolerance(c):
    r0 = c['+0.00']['rate']; ok = {x: bool(c[f'{x:+.2f}']['rate'] >= 0.9 * r0 and c[f'{-x:+.2f}']['rate'] >= 0.9 * r0) for x in (0.15, 0.30, 0.45, 0.90)} if r0 else {}
    return dict(tolerance_m=max([x for x, v in ok.items() if v], default=0.0), holds_at=ok, rate_at_0=r0)
def right(r, dn, tw='canon', n_e=0): return r['R'][dn][tw][n_e]['outcome'] == 'right plant'
P2 = [r for r in lines if r['part'] == 2]; ev = [r for r in P2 if 'skip' not in r]; fail = set(P['summary']['datum_failures']); S = dict(design=P['summary']['design'], twin=P['twin'], revisit=P['rev'], prep={k: v for k, v in P['summary'].items() if k != 'lanes'})
sub = dict(all=ev, without_datum_failure_lanes=[r for r in ev if r['lane'] not in fail], datum_failure_lanes=[r for r in ev if r['lane'] in fail], naming_confirmed=[r for r in ev if r['naming_vote'] == 'agree'], naming_not_confirmed=[r for r in ev if r['naming_vote'] != 'agree'],
           growth_stable=[r for r in ev if r['growth'] == 'stable'], growth_changed=[r for r in ev if r['growth'] in ('grown', 'shrunk')], growth_no_call=[r for r in ev if r['growth'] not in ('stable', 'grown', 'shrunk')])
D2 = dict(n_pairs=len(P2), evaluable=len(ev), not_evaluable=dict(Counter(r['skip'] for r in P2 if 'skip' in r)), truth_readable=P['summary']['truth_readable'], outcomes={}, tolerance={}, geometry_only={}, per_lane={}, pose_offset_px_render_scale={})
for name, R in sub.items(): D2['outcomes'][name] = {dn: {tw: counts(R, dn, tw) for tw in TW} for dn in DAT}
for dn in DAT:
    D2['tolerance'][dn] = {tw: tolerance(D2['outcomes']['all'][dn][tw]) for tw in TW}
    D2['geometry_only'][dn] = {name: dict(n=len(R), partner=sum(r['geometry'][dn]['is_partner'] for r in R), partner_dist_median_m=float(np.median([r['geometry'][dn]['partner_dist_m'] for r in R])) if R else None,
                                          partner_dist_p90_m=float(np.percentile([r['geometry'][dn]['partner_dist_m'] for r in R], 90)) if R else None) for name, R in sub.items() if name in ('all', 'without_datum_failure_lanes', 'datum_failure_lanes')}
    D2['pose_offset_px_render_scale'][dn] = {}
    for tw in TW:
        for which, sel in (('right_plant', lambda x: x['outcome'] == 'right plant'), ('any_pick', lambda x: 'off_px' in x)):
            off = np.array([r['R'][dn][tw][0]['off_px'] for r in ev if sel(r['R'][dn][tw][0])]).reshape(-1, 2)
            if len(off): D2['pose_offset_px_render_scale'][dn][f'{tw}_{which}'] = dict(n=len(off), median_norm=float(np.median(np.linalg.norm(off, axis=1))), p90_norm=float(np.percentile(np.linalg.norm(off, axis=1), 90)), median_du=float(np.median(off[:, 0])), median_dv=float(np.median(off[:, 1])))
for l in sorted({r['lane'] for r in ev}):
    R = [r for r in ev if r['lane'] == l]; D2['per_lane'][str(l)] = dict(n=len(R), datum_failure=bool(l in fail), row_end_minus_oracle_m=LN[l]['row_end_minus_oracle_m'], right_plant_e0={dn: {tw: sum(right(r, dn, tw) for r in R) for tw in TW} for dn in DAT},
                                                                         geometry_partner={dn: sum(r['geometry'][dn]['is_partner'] for r in R) for dn in DAT})
def primary(R, dn='primary', tw='canon'):
    k = sum(right(r, dn, tw) for r in R); n = len(R)
    return dict(n=n, right_plant=k, rate=k / n, ci95_exact=ci(k, n), holds_on_logs=bool(ci(k, n)[0] > 0.80)) if n else dict(n=0)
D2['primary_outcome'] = dict(rule='right-plant rate of rule R at e = 0, primary datum, canon twin, over the evaluable pairs; holds on logs if the lower bound of the exact 95 % interval is above 0.80',
                             all_lanes=primary(ev), without_datum_failure_lanes=primary(sub['without_datum_failure_lanes']), datum_failure_lanes=sorted(fail),
                             beside={f'{dn}_{tw}': dict(all_lanes=primary(ev, dn, tw), without_datum_failure_lanes=primary(sub['without_datum_failure_lanes'], dn, tw)) for dn in DAT for tw in TW if (dn, tw) != ('primary', 'canon')})
D2['paired_e0'] = {}
for x, y in [(x, y) for y in DAT if y in ('oracle', 'sitewide') for x in DAT if x != y]:
    for tw in TW:
        for name in ('all', 'without_datum_failure_lanes'):
            R = sub[name]; g = sum(right(r, x, tw) and not right(r, y, tw) for r in R); b = sum(right(r, y, tw) and not right(r, x, tw) for r in R)
            D2['paired_e0'][f'{x}_vs_{y}_{tw}_{name}'] = {'n': len(R), f'right_only_{x}': g, f'right_only_{y}': b, 'both': sum(right(r, x, tw) and right(r, y, tw) for r in R), 'sign_test_p': sign(g, b)}
S['part2'] = D2
if P['part1']:
    P1 = [r for r in lines if r['part'] == 1]; D1 = dict(n_targets=len(P1), census_inner=sum(r['census_inner'] for r in P1), poses={}, paired={}, gauss={}); med = lambda xs: float(np.median(xs)) if len(xs) else None
    for tw in TW:
        for name, R in (('all', P1), ('census_inner', [r for r in P1 if r['census_inner']])):
            for tag in ('C', 'L', 'R', 'H', 'T'):
                Q = [r['poses'][tag] for r in R if tag in r['poses'] and r['poses'][tag].get(tw) is not None]; lit = [q[tw] for q in Q if q[tw]['lit']]
                D1['poses'][f'{tw}_{name}_{tag}'] = dict(n=len(Q), lit=len(lit), lands=sum(x['lands'] for x in lit), area_px_median=med([x['area_px'] for x in lit]), err_m_median=med([x['err_m'] for x in lit]), err_m_p90=float(np.percentile([x['err_m'] for x in lit], 90)) if lit else None,
                                                         range_m_median=med([x['range_m'] for x in lit]), abs_bearing_deg_median=med([abs(x['bearing_deg']) for x in lit]), train_dist_m_median=med([q['train_dist_m'] for q in Q]), train_dist_m_p90=float(np.percentile([q['train_dist_m'] for q in Q], 90)) if Q else None,
                                                         train_heading_deg_median=med([q['train_heading_deg'] for q in Q]), station_miss_m_median=med([abs(q['station_miss_m']) for q in Q if 'station_miss_m' in q]))
            for x, y in (('C', 'H'), ('C', 'T'), ('L', 'H'), ('R', 'H'), ('H', 'T')):
                B = [r for r in R if x in r['poses'] and y in r['poses'] and r['poses'][x].get(tw) is not None and r['poses'][y].get(tw) is not None]; lx = [bool(r['poses'][x][tw].get('lands')) for r in B]; ly = [bool(r['poses'][y][tw].get('lands')) for r in B]
                g = sum(p and not q for p, q in zip(lx, ly)); b = sum(q and not p for p, q in zip(lx, ly)); D1['paired'][f'{tw}_{name}_{x}_vs_{y}'] = {'n': len(B), f'lands_{x}': sum(lx), f'lands_{y}': sum(ly), f'only_{x}': g, f'only_{y}': b, 'sign_test_p': sign(g, b)}
        gg = [r['gauss'][tw] for r in P1 if tw in r['gauss']]; has = [x for x in gg if x['n'] > 0]
        D1['gauss'][tw] = dict(n=len(gg), with_gaussians=len(has), n_gaussians_median=med([x['n'] for x in has]), dist_ground_m_median=med([x['dist_ground_m'] for x in has]), dist_ground_m_p90=float(np.percentile([x['dist_ground_m'] for x in has], 90)) if has else None, within_half_spacing=sum(x['dist_ground_m'] <= P['summary']['half_spacing_m'] for x in has))
    S['part1'] = D1
json.dump(S, open(OUT / f'rr_summary{a.suffix}.json', 'w'), indent=1)
print(f"== {P['twin']} -> {P['rev']}: {len(ev)} of {len(P2)} evaluable; not evaluable {D2['not_evaluable']}; truth readable {D2['truth_readable']}; datum failure lanes {sorted(fail)}")
for dn in DAT:
    for tw in TW:
        c = D2['outcomes']['all'][dn][tw]; print(f"   {dn:9s} {tw:5s} " + ' | '.join(f"{e} {c[e]['right plant']}/{c[e]['wrong plant']}/{c[e]['unnamed']}/{c[e]['no pick']}" for e in sorted(c, key=float)) + f"  tolerance {D2['tolerance'][dn][tw]['tolerance_m']}")
print('   (e: right / wrong / unnamed / no pick)'); print('   primary outcome:', json.dumps(D2['primary_outcome']['all_lanes']), 'without datum-failure lanes:', json.dumps(D2['primary_outcome']['without_datum_failure_lanes']))
for k, v in D2['paired_e0'].items(): print('   paired', k, v)
print('   geometry only:', json.dumps({dn: D2['geometry_only'][dn]['all'] for dn in DAT})); print('   pose offset px:', json.dumps(D2['pose_offset_px_render_scale']))
if P['part1']:
    for k, v in S['part1']['poses'].items(): print('   part 1', k, json.dumps(v))
    for k, v in S['part1']['paired'].items(): print('   part 1 paired', k, v)
    print('   part 1 gauss', json.dumps(S['part1']['gauss']))
