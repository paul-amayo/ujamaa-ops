#!/usr/bin/env python3
"""twin_ledger_posthoc.py (11 Nov 2026, A100 session) - JIRS test JS-1: POST HOC, NOT A RULE. Written after the scores were seen,
to say what limits the reading; nothing here changes a verdict. From the records and prep.json only:
  1. T-id4 with the lidar's lane constant in both twins (the design's secondary height line, which the design applies to T-id3
     only) against the paper's call;
  2. T-id1 decomposed: twin-read position against the anchor of its identity (does the field put the address at its anchor?),
     the anchor against the census bag (the naming offset), and readings far (> 2 m) from both;
  3. the lane constants per cell.
  python3 twin_ledger_posthoc.py <prep.json> <records dir> <results dir>"""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
P = json.load(open(sys.argv[1])); REC, RES = Path(sys.argv[2]), Path(sys.argv[3]); SUM = json.load(open(RES / 'summary.json')); TH = 0.082
CALLS = ['grown', 'stable', 'shrunk']; lab = lambda c: 'grown' if c > TH else ('shrunk' if c < -TH else 'stable')
plants = {s: {e['address']: e for e in (json.loads(l) for l in open(REC / f'plants_{s}.jsonl')) if e['in_vocab']} for s in ('dec', 'apr')}
pairs = [json.loads(l) for l in open(REC / 'pairs.jsonl')]; out = {'note': 'post hoc, not a rule: computed after the scores were seen; no verdict depends on it'}
# 1. T-id4 with the lidar lane constants
t4 = {}
for name, sel in (('all_twin_pairs_with_paper_call', lambda p: True), ('same_pairs_of_T_id2', lambda p: p['category'] == 'same pair')):
    M = np.zeros((3, 3), int); ch = []
    for p in pairs:
        if p['kind'] != 'twin_pair' or p['dec_bag'] is None or p['L2'] is None or not sel(p): continue
        hd, ha = plants['dec'][p['A_dec']]['b'].get('height_lidarlane'), plants['apr'][p['A_apr']]['b'].get('height_lidarlane')
        if hd is None or ha is None: continue
        c = ha - hd; ch.append(c); M[CALLS.index(lab(c)), CALLS.index(p['L2'])] += 1
    t4[name] = dict(n=int(M.sum()), matrix=M.tolist(), agree=int(np.trace(M)), agreement=float(np.trace(M) / M.sum()) if M.sum() else None, change_median=float(np.median(ch)) if ch else None)
out['T_id4_lidar_lane_constants'] = t4
# 2. T-id1 decomposed against the anchors
t1 = {}
for s in ('dec', 'apr'):
    anc = {int(k): np.array(v, float) for k, v in P[s]['anchors_graph'].items()}; rows = []
    for a, e in plants[s].items():
        b = e.get('named_bag')
        if b is None: continue
        A = anc[a]; cz = e['census_xyz']; ad = float(np.hypot(A[0] - cz[0], A[2] - cz[2]))
        pos = e['b'].get('pos'); td = float(np.hypot(pos[0] - cz[0], pos[2] - cz[2])) if pos else None; ta = float(np.hypot(pos[0] - A[0], pos[2] - A[2])) if pos else None
        rows.append((ad, td, ta))
    read = [r for r in rows if r[1] is not None]
    t1[s] = dict(named=len(rows), read=len(read), anchor_to_census_median=float(np.median([r[0] for r in rows])), anchor_to_census_p95=float(np.percentile([r[0] for r in rows], 95)),
                 twin_within_0_5_of_census=sum(r[1] <= 0.5 for r in read), twin_within_0_5_of_anchor=sum(r[2] <= 0.5 for r in read),
                 near_anchor_but_beyond_0_5_of_census=sum(r[2] <= 0.5 and r[1] > 0.5 for r in read), beyond_2_m_of_census=sum(r[1] > 2.0 for r in read), beyond_2_m_of_anchor=sum(r[2] > 2.0 for r in read),
                 between_0_5_and_2_m_of_census=sum(0.5 < r[1] <= 2.0 for r in read), twin_to_anchor_median=float(np.median([r[2] for r in read])))
out['T_id1_against_anchors'] = t1
# 3. lane constants per cell
out['lane_constant_per_cell'] = {s: SUM['lane_constant'][s]['per_cell'] for s in ('dec', 'apr')}
json.dump(out, open(RES / 'posthoc.json', 'w'), indent=1); print(json.dumps(out, indent=1))
