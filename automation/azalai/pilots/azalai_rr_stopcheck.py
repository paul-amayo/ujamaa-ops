#!/usr/bin/env python3
"""azalai_rr_stopcheck.py (11 Nov 2026, A100 session) — independent check of the truth-free stop-frame choice and the believed pose of
Part 2 (RR_design.md, steps 1 to 3), re-derived from the stored lane frames, the stpg graph and the census with separately written
arithmetic: the target's station in the twin's primary lane frame, the stop station (minus d in the driving sense), the revisit
keyframe of the lane whose station (revisit lane frame of the datum) is nearest; and, for the e = 0 believed pose (F_twin F_rev^-1
applied to the stop frame's pose, then the graph-to-cell rigid map), its distance and rotation from the cell pose prep gave the
renderer, which may differ only by the anchor keyframe's own residual (H3: 0.038 m median over keyframes). Exit status 1 on any
disagreement of a stop frame or a station. nerf_new python3.10:
  azalai_rr_stopcheck.py <prep.json>"""
import sys, json, numpy as np
P = json.load(open(sys.argv[1])); E = json.load(open(P['stpg']))['experiences']; Cr = np.array(E[P['rev']]['c2w'], float)
Bt = np.load(P['census_twin'])['bags_xyz'].astype(float); d = P['summary']['standoff_d_m']; LN = {l['lane']: l for l in P['lanes']}; RP = P['rev_passes']
TR = np.array(P['T_graph_to_cell']['R']); Tt = np.array(P['T_graph_to_cell']['t']); jobs = {j['job']: j for j in P['jobs']}; bad = 0; n = 0; dev = []; rot = []
for t in P['targets']:
    if 'skip' in t: continue
    L = LN[t['lane']]; Ft = np.array(L['F_twin']); sn = L['sense']; a = Ft[:3, 2]; o = Ft[:3, 3]
    s_t = float(np.dot(Bt[t['twin_bag']] - o, a)); s_stop = s_t - d * sn
    if abs(s_t - t['station_m']) > 1e-9 or abs(s_stop - t['stop_station_m']) > 1e-9: bad += 1; print('station differs', t['twin_bag'])
    k0, k1, _ = RP[t['lane'] - 1]
    for dn, rec in t['datum'].items():
        if dn == 'sitewide': continue
        Fr = np.array(L['F_rev'][dn]); st = np.array([np.dot(Cr[k, :3, 3] - Fr[:3, 3], Fr[:3, 2]) for k in range(k0, k1 + 1)])
        k = k0 + int(np.argmin(np.abs(st - s_stop))); n += 1
        if k != rec['stop_k']: bad += 1; print('stop frame differs', t['twin_bag'], dn, k, rec['stop_k'])
        Pb = Ft @ np.linalg.inv(Fr) @ Cr[rec['stop_k']]; q = np.array(jobs[rec['jobs'][0]]['c2w'])
        dev.append(float(np.linalg.norm((TR @ Pb[:3, 3] + Tt) - q[:3, 3]))); R = (TR @ Pb[:3, :3]).T @ q[:3, :3]; rot.append(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
dev = np.array(dev); rot = np.array(rot)
print(f'{n} target x datum stop frames re-derived; disagreements {bad}. Believed pose (rigid map) against the rendered cell pose: position median {np.median(dev):.3f} m, p90 {np.percentile(dev, 90):.3f}, max {dev.max():.3f}; rotation median {np.median(rot):.2f} deg, p90 {np.percentile(rot, 90):.2f}, max {rot.max():.2f}')
sys.exit(1 if bad else 0)
