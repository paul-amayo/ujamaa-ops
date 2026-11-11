#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3, 2026-11-11), step 3: the laser census plants of the rig day projected into every dumped view.

Chain (all from existing records, nothing fitted here):
  laser census of the date (lmi_<date>.json plants: x, y, ground_z, height_p95, row; laser frame, laser datum)
  -> the splat's registration reference frame: + (dE, dN)[laser_<date>] - (dE, dN)[<reference sensor>_<reference date>] of frames.json
  -> the 3D point of a plant = (x, y, ground_z + 0.5 * height_p95)   [the middle of the plant: a judgement, named in the design]
  -> rig metres by the inverse of registration.json[stop].T_rig_to_utm_datum_corrected (the splat's z is on the laser datum)
  -> model units by the run's dataparser transform and scale -> pixel by the view's pinhole K (cameras.json, OpenGL camera).
Writes <stop dir>/plants.json: per plant id, row, on_row, the point, and per view {u, v, depth_m, in_image}.
Either box python (numpy only):  mi_project.py --stop 230613_pos00 --dir <stop dir> [--date 230613]"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--stop', required=True); ap.add_argument('--dir', required=True); ap.add_argument('--date', default=None)
ap.add_argument('--census', default='/mnt/shared2/must-c/plot198/census'); a = ap.parse_args()
date = a.date or a.stop.split('_')[0]
J = json.load(open(f'{a.census}/lmi_{date}.json')); F = json.load(open(f'{a.census}/frames.json'))['frames']; R = json.load(open(f'{a.census}/registration.json'))[a.stop]
C = json.load(open(f'{a.dir}/cameras.json'))
ref = R['reference']; kl, kr = f'laser_{date}', f"{ref['kind']}_{ref['date']}"
dE = F[kl]['dE_m'] - F[kr]['dE_m']; dN = F[kl]['dN_m'] - F[kr]['dN_m']
T = np.array(R['T_rig_to_utm_datum_corrected']); Ti = np.linalg.inv(T)
Tdp = np.array(C['dataparser_transform']); s = C['dataparser_scale']
soil = np.array(R['footprint_soil_polygon_utm'])
def in_poly(poly, x, y):
    n = len(poly); c = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: c = not c
    return c
out = {'stop': a.stop, 'date': date, 'laser_census': f'{a.census}/lmi_{date}.json', 'reference': kr, 'laser_to_reference_shift_en_m': [dE, dN],
       'frames_json_entries': {kl: F[kl], kr: F[kr]}, 'T_rig_to_utm_datum_corrected': T.tolist(), 'datum_correction_m': R['datum_correction_m'],
       'point_rule': 'x + dE, y + dN, ground_z + 0.5 * height_p95', 'plants': []}
for p in J['plants']:
    P = np.array([p['x'] + dE, p['y'] + dN, p['ground_z'] + 0.5 * p['height_p95'], 1.0])
    rig = Ti @ P; mod = (Tdp[:, :3] @ rig[:3] + Tdp[:, 3]) * s
    rec = {'id': int(p['id']), 'row': int(p['row']), 'on_row': bool(p['row'] >= 0), 'h95_m': p['height_p95'], 'area_cm2': p['area_cm2'], 'xyz_ref': P[:3].tolist(), 'xyz_rig_m': rig[:3].tolist(),
           'in_soil_polygon': bool(in_poly(soil, P[0], P[1])), 'views': {}}
    for name, c in C['frames'].items():
        c2w = np.array(c['c2w_model']); K = np.array(c['K']); pc = c2w[:, :3].T @ (mod - c2w[:, 3])
        if pc[2] >= -1e-6: rec['views'][name] = {'in_image': False, 'behind': True}; continue
        u = K[0, 0] * pc[0] / -pc[2] + K[0, 2]; v = K[1, 1] * -pc[1] / -pc[2] + K[1, 2]
        rec['views'][name] = {'u': float(u), 'v': float(v), 'depth_m': float(-pc[2] / s), 'in_image': bool(0 <= u < c['w'] and 0 <= v < c['h'])}
    out['plants'].append(rec)
json.dump(out, open(f'{a.dir}/plants.json', 'w'), indent=1)
n = len(out['plants']); on = sum(p['on_row'] for p in out['plants'])
vis = {name: sum(p['views'][name].get('in_image', False) for p in out['plants']) for name in C['frames']}
print(f'[project] {a.stop}: {n} laser instances ({on} on a row), shift laser->{kr} E {dE:+.4f} N {dN:+.4f} m; in soil polygon {sum(p["in_soil_polygon"] for p in out["plants"])}; in image per view: {vis}')
