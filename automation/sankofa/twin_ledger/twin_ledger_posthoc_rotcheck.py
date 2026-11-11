# geometry only: relative rotation between each cell camera (mapped to the graph frame by R^T) and the graph camera of the same keyframe
import json, numpy as np
P = json.load(open('/mnt/shared/sankofa_twin/prep.json'))
E = json.load(open(P['stpg']))['experiences']
for s, pas in (('dec', 'dec'), ('apr', 'apr1054')):
    Cg = np.array(E[pas]['c2w'], float); R = np.array(P[s]['R']); up = np.array(P[s]['up'])
    for cell, C in sorted(P[s]['cells'].items()):
        ang, pitch, fwd_up_cell, fwd_up_graph = [], [], [], []
        for f in C['frames']:
            k = f['k']; Rc = R.T @ np.array(f['c2w'])[:3, :3]; Rg = Cg[k][:3, :3]
            D = Rg.T @ Rc; ang.append(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))))
            fwd_up_cell.append(np.degrees(np.arcsin(np.clip(Rc[:, 2] @ up, -1, 1)))); fwd_up_graph.append(np.degrees(np.arcsin(np.clip(Rg[:, 2] @ up, -1, 1))))
        print(f'{s} {cell}: rel. rotation cell vs graph median {np.median(ang):.2f} deg (p90 {np.percentile(ang, 90):.2f}); optical-axis elevation cell {np.median(fwd_up_cell):+.2f} deg, graph {np.median(fwd_up_graph):+.2f} deg; K {C["K"]}')
