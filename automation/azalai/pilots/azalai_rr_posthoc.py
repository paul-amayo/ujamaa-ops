#!/usr/bin/env python3
"""azalai_rr_posthoc.py (11 Nov 2026, A100 session) — POST HOC, NOT A RULE. Diagnostics of rule R's naming step on one rendered
rung, written after the pair of record's outcomes were read, to explain them; nothing here changes an outcome of record.
  (1) naming geometry: at every evaluable target's stop frame (each datum, e = 0) the partner's census point projected with the
      rig intrinsics (the rule's) and with the cell intrinsics (the render's): in the image or not, inside m* or not, inside any
      live mask or not; for all census bags within 4 m of the camera and in view, the fraction that falls inside some live mask
      (a hit rate per intrinsics set, which says which set puts the census on the bags);
  (2) the SENSITIVITY the Mac agent named before scoring (STATE.md, decision 9): rule R's naming with the cell intrinsics instead
      of the rig intrinsics, every datum, twin and e, from the stored m* (same masks, same M_t, same pick); labelled as a
      sensitivity, never adopted;
  (3) a breakdown of the outcomes at e = 0 by the target's distance from the nearer row end of its lane (twin lane frame), and by
      whether the partner's census point is in the image.
Writes rr_posthoc<suffix>.json. nerf_new python3.10:
  azalai_rr_posthoc.py <prep.json> <rr_targets<suffix>.jsonl> <out dir> --suffix _dec_apr1130"""
import argparse, json, glob, sys, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import pycocotools.mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths
from rig_calib import load_rig
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('targets'); ap.add_argument('out'); ap.add_argument('--suffix', default=''); a = ap.parse_args()
P = json.load(open(a.prep)); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); ES = P['summary']['sweep_m']; DAT = P['summary']['datums']
E = json.load(open(P['stpg']))['experiences']; Cr = np.array(E[P['rev']]['c2w'], float); Br = np.load(P['census_rev'])['bags_xyz'].astype(float); finr = np.flatnonzero(np.isfinite(Br).all(1))
KR = load_rig(survey_paths.monos(P['rev_root']))['intrinsics']; KC = dict(zip(('fx', 'fy', 'cx', 'cy'), next(iter(P['cells'].values()))['K']))
assert all(c['K'] == next(iter(P['cells'].values()))['K'] for c in P['cells'].values()), 'cells differ in intrinsics'
KS = dict(rig=KR, cell=KC)
ENT = {}
for f in sorted(glob.glob(str(Path(P['rev_root']) / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
    c = json.load(open(f))
    for li, fr in enumerate(c['frames']): ENT.setdefault(int(fr['kf_idx']), [x['rle'] for x in c['frame_entries'].get(str(li), [])])
cache = {}
def live(k):
    if k not in cache:
        ms = []
        for rle in ENT.get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool))
        if len(cache) > 200: cache.clear()
        cache[k] = np.stack(ms) if ms else np.zeros((0, 720, 1280), bool)
    return cache[k]
def project(k, K, idx):
    w2c = np.linalg.inv(Cr[k]); q = (w2c[:3, :3] @ Br[idx].T).T + w2c[:3, 3]; ok = q[:, 2] > 0; u = np.full(len(q), -1.0); v = np.full(len(q), -1.0)
    u[ok] = K['fx'] * q[ok, 0] / q[ok, 2] + K['cx']; v[ok] = K['fy'] * q[ok, 1] / q[ok, 2] + K['cy']; inb = ok & (u >= 0) & (u < 1280) & (v >= 0) & (v < 720); return q, u, v, inb
def name_with(k, i, K):
    M = live(k); q, u, v, inb = project(k, K, finr); ins = np.flatnonzero(inb); ins = ins[M[i][v[ins].astype(int), u[ins].astype(int)]]
    return int(finr[ins[int(np.argmin(np.linalg.norm(q[ins], axis=1)))]]) if len(ins) else None
L = [json.loads(l) for l in open(a.targets)]; ev = [r for r in L if r['part'] == 2 and 'skip' not in r]; TW = [tw for tw in ('canon', 'bag') if tw in ev[0]['R'][DAT[0]]]
PT = {p['twin_bag']: p for p in P['targets']}; LN = {l['lane']: l for l in P['lanes']}
O = dict(note='POST HOC, NOT A RULE: diagnostics and the sensitivity named in STATE.md decision 9 (cell intrinsics for the naming); never adopted', rig_K=KR, cell_K=KC)
# (1) naming geometry
geo = defaultdict(Counter); hit = {kn: [0, 0] for kn in KS}; seen_frames = set()
for r in ev:
    for dn in DAT:
        k = r['R'][dn]['stop_k']; x = r['R'][dn]['canon'][0]
        for kn, K in KS.items():
            q, u, v, inb = project(k, K, np.array([r['partner']])); M = live(k)
            st = 'off image' if not inb[0] else ('inside m*' if (x.get('mstar') is not None and M[x['mstar']][int(v[0]), int(u[0])]) else ('inside another live mask' if M[:, int(v[0]), int(u[0])].any() else 'in no live mask'))
            geo[f'{dn}_{kn}'][f"{x['outcome']} | partner point {st}"] += 1
        if (k not in seen_frames) and dn == 'oracle':
            seen_frames.add(k); M = live(k)
            for kn, K in KS.items():
                q, u, v, inb = project(k, K, finr); sel = np.flatnonzero(inb & (np.linalg.norm(q, axis=1) < 4.0))
                hit[kn][0] += int(M[:, v[sel].astype(int), u[sel].astype(int)].any(0).sum()) if len(M) else 0; hit[kn][1] += len(sel)
O['naming_geometry_e0_canon'] = {k: dict(v) for k, v in geo.items()}
O['census_hit_rate_oracle_stop_frames'] = {kn: dict(inside_some_live_mask=h[0], census_points_within_4m_in_view=h[1], rate=h[0] / h[1] if h[1] else None, frames=len(seen_frames)) for kn, h in hit.items()}
# (2) the sensitivity: naming with the cell intrinsics, from the stored m*
sens = {}
for dn in DAT:
    for tw in TW:
        c = {e: Counter() for e in ES}
        for r in ev:
            k = r['R'][dn]['stop_k']
            for n_e, e in enumerate(ES):
                x = r['R'][dn][tw][n_e]
                if x['outcome'] == 'no pick': c[e]['no pick'] += 1; continue
                nm = name_with(k, x['mstar'], KC); c[e]['unnamed' if nm is None else ('right plant' if nm == r['partner'] else 'wrong plant')] += 1
        sens[f'{dn}_{tw}'] = {f'{e:+.2f}': dict(n=len(ev), **{o: c[e].get(o, 0) for o in ('right plant', 'wrong plant', 'unnamed', 'no pick')}) for e in ES}
O['sensitivity_cell_intrinsics_naming'] = sens
# a recount check of the rule's own naming (rig intrinsics) by this file's arithmetic at e = 0, primary datum, canon
mism = sum((name_with(r['R']['primary']['stop_k'], r['R']['primary']['canon'][0]['mstar'], KR) != r['R']['primary']['canon'][0]['name']) for r in ev if r['R']['primary']['canon'][0]['outcome'] != 'no pick')
O['rig_naming_recount_mismatches_primary_canon_e0'] = mism
# (3) outcomes at e = 0 by distance from the nearer row end (twin lane frame, from the twin's inner-row stations) and by partner in view
rows = defaultdict(Counter)
for r in ev:
    l = LN[r['lane']]; length = l['twin_inner']['length_m']; s = abs(r['station_m']); dend = min(s, abs(length - s)); band = '< 3 m' if dend < 3 else ('3-6 m' if dend < 6 else '>= 6 m')
    for dn in ('primary', 'oracle'):
        rows[f'{dn}_canon_row_end_band'][f"{band}: {r['R'][dn]['canon'][0]['outcome']}"] += 1
O['outcome_by_row_end_distance_e0'] = {k: dict(sorted(v.items())) for k, v in rows.items()}
json.dump(O, open(OUT / f'rr_posthoc{a.suffix}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in O.items() if k not in ('sensitivity_cell_intrinsics_naming',)}, indent=1))
for k, v in sens.items(): print('sensitivity (cell intrinsics naming)', k, ' | '.join(f"{e} {x['right plant']}/{x['wrong plant']}/{x['unnamed']}/{x['no pick']}" for e, x in sorted(v.items(), key=lambda t: float(t[0]))))
