#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3, 2026-11-11), measurement (ii) of the design of record: per-plant height read through the
address, against the laser, beside the address-free readings on the same plants.

  points     the stop's method b points (mustc_splat_geom.py: back-projected rendered depth of the training cameras, datum-corrected
             heights above the 230601 laser DEM) with keep true and inside footprint_soil_polygon_utm = the set the splat census reads
  address    a point is projected back into its source training camera (the pinhole it came from; the recovery is checked against the
             stride-2 pixel-centre lattice) and takes the plant-level argmax of that pixel (argmax_eval.py --dump-pred, key 'tree';
             -1 = void) at the render scale of the maps
  h95        95th percentile of the heights of the points carrying a plant's address, >= 20 points (census minimum), else not found
  S          the on-row laser instances (nodes) inside the stop's soil polygon (plants.json)
  c1         the address-free splat census method b (<stop>__b.json) matched to the laser instances by the ledger's cross-sensor rule
             on this stop alone: mutual nearest neighbour within 0.10 m among ALL laser instances and ALL splat instances, the median
             residual translation of the first pass removed if >= 5 pairs, match repeated (mustc_ledger.xs_match / mutual_nn)
  c2         the record's laser-footprint h95 (dsm_vs_laser.json, nview6, method b, per plant)
Writes <dir>/heights.csv (one row per plant of S), heights_summary.json, address_points.json.
Box python with numpy + scipy + matplotlib (the must-c venv or nerf_new):  mi_heights.py --stop 230613_pos00 --dir <stop dir>"""
import argparse, csv, json
import numpy as np
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('--stop', required=True); ap.add_argument('--dir', required=True)
ap.add_argument('--census', default='/mnt/shared2/must-c/plot198/census'); ap.add_argument('--geom', default='/mnt/shared/must-c_bulk/work/splat_geom')
ap.add_argument('--ws', default='/mnt/shared/must-c_bulk/work/ws'); ap.add_argument('--pred', default='pred_tangent_vf'); ap.add_argument('--min-pts', type=int, default=20); ap.add_argument('--r', type=float, default=0.10)
a = ap.parse_args(); D = a.dir
P = json.load(open(f'{D}/plants.json')); C = json.load(open(f'{D}/cameras.json')); plants = P['plants']
z = np.load(f'{a.geom}/{a.stop}/depth_points_utm.npz'); xyz = z['xyz']; h = z['h'].astype(np.float64); keep = z['keep']; src = z['src_cam'].astype(int); poly = np.asarray(z['footprint_soil_polygon_utm'])
T = np.asarray(z['T_rig_to_utm_datum_corrected']); assert np.abs(T - np.array(P['T_rig_to_utm_datum_corrected'])).max() < 1e-9, 'registration of the points differs from the one the census was projected with'
def in_poly(poly, x, y):
    n = len(poly); c = np.zeros(len(x), bool)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        with np.errstate(divide='ignore', invalid='ignore'):
            c ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return c
inp = in_poly(poly, xyz[:, 0], xyz[:, 1]); use = keep & inp
TR = json.load(open(f'{a.ws}/{a.stop}/transforms_ba.json')); split = json.load(open(f'{a.ws}/{a.stop}/split.json'))
cam_labels = [fr['camera_label'] for fr in TR['frames'] if fr['camera_label'] in split['train']]   # the order mustc_splat_geom.py built its cameras in (src_cam index)
by_label = {c['label']: (n, c) for n, c in C['frames'].items()}
Ti = np.linalg.inv(T); Tdp = np.array(C['dataparser_transform']); s = C['dataparser_scale']
rig = xyz @ Ti[:3, :3].T + Ti[:3, 3]; mod = (rig @ Tdp[:, :3].T + Tdp[:, 3]) * s
addr = np.full(len(xyz), -2, np.int64)   # -2 = not evaluated, -1 = void
lat = {}
for ci, label in enumerate(cam_labels):
    sel = np.flatnonzero(src == ci)
    if not len(sel): continue
    name, c = by_label[label]; c2w = np.array(c['c2w_model']); K = np.array(c['K']); pc = (mod[sel] - c2w[:, 3]) @ c2w[:, :3]
    u = K[0, 0] * pc[:, 0] / -pc[:, 2] + K[0, 2]; v = K[1, 1] * -pc[:, 1] / -pc[:, 2] + K[1, 2]
    du = np.abs((u - 0.5) / 2 - np.round((u - 0.5) / 2)) * 2; dv = np.abs((v - 0.5) / 2 - np.round((v - 0.5) / 2)) * 2   # distance to the stride-2 pixel-centre lattice, px
    lat[label] = {'n': int(len(sel)), 'lattice_resid_px_median': float(np.median(np.hypot(du, dv))), 'lattice_resid_px_p99': float(np.percentile(np.hypot(du, dv), 99)),
                  'depth_m_median': float(np.median(-pc[:, 2] / s))}
    pr = np.load(f'{D}/{a.pred}/{name}.npz'); tree = pr['tree']; Hh, Ww = tree.shape; sx, sy = Ww / c['w'], Hh / c['h']
    iu = np.clip(np.floor(u * sx).astype(int), 0, Ww - 1); iv = np.clip(np.floor(v * sy).astype(int), 0, Hh - 1)
    addr[sel] = tree[iv, iu]
assert (addr[use] > -2).all(), 'a used point has a source camera outside the training list'
ids_bank = sorted(int(i) for i in np.unique(addr[use]) if i >= 0)
apts = {}
for i in ids_bank:
    m = use & (addr == i); apts[i] = {'n_points': int(m.sum()), 'h95_m': float(np.percentile(h[m], 95)) if m.sum() >= a.min_pts else None,
                                      'xy_median': [float(np.median(xyz[m, 0])), float(np.median(xyz[m, 1]))], 'n_source_cameras': int(len(np.unique(src[m])))}
# c1: the address-free splat census, ledger rule on this stop
SC = json.load(open(f'{a.census}/splat_census/{a.stop}__b.json')); inst = SC['instances']
sx_ = np.array([i['x_utm'] for i in inst]); sy_ = np.array([i['y_utm'] for i in inst]); sh = np.array([i['h95_m'] for i in inst])
lx = np.array([p['xyz_ref'][0] for p in plants]); ly = np.array([p['xyz_ref'][1] for p in plants])
def mutual_nn(ax, ay, bx, by, max_d):
    ta, tb = cKDTree(np.column_stack([ax, ay])), cKDTree(np.column_stack([bx, by]))
    dab, jab = ta.query(np.column_stack([bx, by]), k=1, distance_upper_bound=max_d); dba, jba = tb.query(np.column_stack([ax, ay]), k=1, distance_upper_bound=max_d)
    ia = np.arange(len(ax)); okb = np.isfinite(dba); jj = np.where(okb, jba, 0); ok = okb & (jab[jj] == ia)
    return ia[ok], jba[ok], dba[ok]
# A100 session, 11-11 20:15 UTC, before any score: the laser set is xs_match's A["in_region"] = the ledger's LASER_BAND (raw laser UTM)
# minus the stake-rule ids of the date (mustc_ledger.load_laser); the splat set is every instance of this stop (the ledger's splat in_region).
# The Mac version matched against all laser instances; that variant is kept below only as a labelled check (c1_check_all_laser_instances).
LASER_BAND = {'x': (357415.6, 357421.7), 'y': (5610200.3, 5610201.95)}   # mustc_ledger.py line 159
date = a.stop.split('_')[0]; raw = {int(p['id']): p for p in json.load(open(f'{a.census}/lmi_{date}.json'))['plants']}
stakes = set(int(i) for i in json.load(open(f'{a.census}/lmi_assoc.json')).get('stake_rule', {}).get('per_date', {}).get(date, {}).get('ids', []))
in_band = np.array([LASER_BAND['x'][0] <= raw[p['id']]['x'] <= LASER_BAND['x'][1] and LASER_BAND['y'][0] <= raw[p['id']]['y'] <= LASER_BAND['y'][1] and p['id'] not in stakes for p in plants])
def xs(sel):   # xs_match on the laser subset sel: mutual NN within r, first-pass median translation removed if >= 5 pairs, match repeated
    sa = np.flatnonzero(sel); iS, iL, d = mutual_nn(sx_, sy_, lx[sa], ly[sa], a.r); n1 = int(len(iS)); tr = None
    if len(iS) >= 5:
        tr = [float(np.median(lx[sa][iL] - sx_[iS])), float(np.median(ly[sa][iL] - sy_[iS]))]; iS, iL, d = mutual_nn(sx_ + tr[0], sy_ + tr[1], lx[sa], ly[sa], a.r)
    return {int(plants[sa[j]]['id']): (int(i), float(dd)) for i, j, dd in zip(iS, iL, d)}, n1, tr
c1, raw_n, tr = xs(in_band)
c1_all, raw_n_all, tr_all = xs(np.ones(len(plants), bool))
# c2: the record's laser-footprint h95
try:
    DV = json.load(open(f'{a.census}/dsm_vs_laser.json'))[a.stop]['vs_laser']['by_footprint']['nview6']['b']['per_plant']
    # A100 session 11-11 20:20 UTC: the record numbers its plants with its own local ids (2, 4, 5, ...), not the lmi census ids; the first run
    # joined by id and found 0 of 11. Join by the raw laser position (<= 1 mm) and an identical laser h95 (<= 1e-6 m) instead; unmatched counted.
    c2, c2_unmatched = {}, []
    for q in DV['plants']:
        hit = [i for i, p in raw.items() if abs(p['x'] - q['x']) <= 1e-3 and abs(p['y'] - q['y']) <= 1e-3 and abs(p['height_p95'] - q['laser_h95']) <= 1e-6]
        if len(hit) == 1: c2[hit[0]] = q
        else: c2_unmatched.append(int(q['id']))
    DV['join'] = {'rule': 'record plant -> lmi id by raw position <= 1 mm and identical laser h95', 'matched': {str(int(q['id'])): i for i, q in c2.items()}, 'unmatched_record_ids': c2_unmatched}
except Exception as e:  # noqa: BLE001
    DV, c2 = {'error': repr(e)[:200]}, {}
sup = {}
for r in csv.DictReader(open(f'{D}/cell/plant_views.csv')):
    if r['outcome'] == 'labelled': sup.setdefault(int(r['plant']), {'train': 0, 'eval': 0})[r['split']] += 1
S = [p for p in plants if p['on_row'] and p['in_soil_polygon']]; rows = []
for p in S:
    i = p['id']; ad = apts.get(i); m1 = c1.get(i); m2 = c2.get(i)
    rows.append({'plant': i, 'row': p['row'], 'laser_h95_m': round(p['h95_m'], 4), 'laser_area_cm2': p['area_cm2'], 'census_views_labelled': sup.get(i, {}).get('train', 0), 'held_out_views_labelled': sup.get(i, {}).get('eval', 0),
                 'address_n_points': ad['n_points'] if ad else 0, 'address_h95_m': round(ad['h95_m'], 4) if ad and ad['h95_m'] is not None else '',
                 'c1_instance': m1[0] if m1 else '', 'c1_dist_m': round(m1[1], 4) if m1 else '', 'c1_h95_m': round(float(sh[m1[0]]), 4) if m1 else '',
                 'c2_h95_m': round(m2['splat_h95'], 4) if m2 and m2.get('splat_h95') is not None else '', 'c2_n_points': m2['n_splat_pts'] if m2 else '', 'c2_in_record': int(m2 is not None)})
with open(f'{D}/heights.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def st(key, sub=None):
    rr = [r for r in rows if r[key] != '' and (sub is None or r['plant'] in sub)]; dh = np.array([r[key] - r['laser_h95_m'] for r in rr])
    if not len(dh): return {'n': 0}
    return {'n': int(len(dh)), 'bias_median_m': float(np.median(dh)), 'absdiff_median_m': float(np.median(np.abs(dh))), 'absdiff_p95_m': float(np.percentile(np.abs(dh), 95)), 'absdiff_max_m': float(np.abs(dh).max()), 'plants': [r['plant'] for r in rr]}
common = {r['plant'] for r in rows if r['address_h95_m'] != '' and r['c1_h95_m'] != '' and r['c2_h95_m'] != ''}
common2 = {r['plant'] for r in rows if r['address_h95_m'] != '' and r['c1_h95_m'] != ''}
summ = {'stop': a.stop, 'pred': a.pred, 'points': {'total': int(len(xyz)), 'keep': int(keep.sum()), 'keep_and_in_soil_polygon': int(use.sum()), 'with_an_address': int((addr[use] >= 0).sum()), 'void': int((addr[use] == -1).sum()),
                                                     'above_0.04_m': int((h[use] > 0.04).sum()), 'above_0.04_m_with_an_address': int(((h > 0.04) & use & (addr >= 0)).sum())},
        'pixel_recovery_check': lat, 'source_cameras': cam_labels, 'min_points': a.min_pts,
        'S': {'n': len(S), 'ids': [p['id'] for p in S], 'with_census_supervision': int(sum(1 for r in rows if r['census_views_labelled'] > 0))},
        'address': {'found': int(sum(1 for r in rows if r['address_h95_m'] != '')), **st('address_h95_m')},
        'c1_splat_census_b': {'found': int(sum(1 for r in rows if r['c1_h95_m'] != '')), **st('c1_h95_m'), 'n_splat_instances': len(inst), 'n_laser_instances': len(plants), 'first_pass_pairs': raw_n, 'translation_applied_to_splat_en_m': tr, 'match_radius_m': a.r,
                              'matched_all_laser_instances': len(c1), 'laser_set': f'xs_match in_region: LASER_BAND and not stake ({int(in_band.sum())} of {len(plants)} instances; excluded ids {[p["id"] for p, b in zip(plants, in_band) if not b]})'},
        'c1_check_all_laser_instances': {'note': 'check only (the Mac version of this script): xs_match against all laser instances, no band', 'first_pass_pairs': raw_n_all, 'translation_applied_to_splat_en_m': tr_all,
                                         'S_plants_whose_match_differs': sorted(int(p['id']) for p in S if c1.get(p['id']) != c1_all.get(p['id']))},
        'c2_laser_footprint_record': {'found': int(sum(1 for r in rows if r['c2_h95_m'] != '')), 'in_record': int(sum(r['c2_in_record'] for r in rows)), **st('c2_h95_m'), 'record_raw': DV.get('raw') if isinstance(DV, dict) else None, 'join': DV.get('join') if isinstance(DV, dict) else None},
        'paired_all_three': {'n': len(common), 'address': st('address_h95_m', common), 'c1': st('c1_h95_m', common), 'c2': st('c2_h95_m', common)},
        'paired_address_c1': {'n': len(common2), 'address': st('address_h95_m', common2), 'c1': st('c1_h95_m', common2)},
        'address_ids_outside_S': {str(i): v for i, v in apts.items() if i not in {p['id'] for p in S}}}
json.dump(summ, open(f'{D}/heights_summary.json', 'w'), indent=1); json.dump({str(k): v for k, v in apts.items()}, open(f'{D}/address_points.json', 'w'), indent=1)
print(f"[heights] points used {use.sum()} ({(addr[use] >= 0).sum()} with an address); pixel recovery median resid {np.median([v['lattice_resid_px_median'] for v in lat.values()]):.4f} px")
for k in ('address', 'c1_splat_census_b', 'c2_laser_footprint_record'):
    v = summ[k]; print(f"[heights] {k}: found {v['found']} of {len(S)}; bias {v.get('bias_median_m')} |dh95| median {v.get('absdiff_median_m')} p95 {v.get('absdiff_p95_m')} (n {v.get('n')})")
