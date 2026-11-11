# post hoc (after scoring): April chunks' word tables and where each chunk reads an address relative to its anchor
import json, numpy as np
from pathlib import Path
P = json.load(open('/mnt/shared/sankofa_twin/prep.json')); S = P['apr']
wts = {}
for c, C in sorted(S['cells'].items()):
    b = np.load(Path(C['identity']) / 'text_bank.npz', allow_pickle=True); m = json.loads(str(b['meta'])); wts[c] = {int(k): v for k, v in m['word_table'].items() if int(k) < 10000}
    print(c, 'ids', len(wts[c]), 'hierarchy', m.get('hierarchy_json'), 'manifest', m.get('manifest'))
cs = sorted(wts); print('same id->word in all chunks:', all(wts[c] == wts[cs[0]] for c in cs))
anc = {int(k): np.array(v) for k, v in S['anchors_graph'].items()}
cam = np.array([v for v in S['cam_centres_graph'].values()]); print('graph camera x range', cam[:, 0].min().round(1), cam[:, 0].max().round(1), 'z range', cam[:, 2].min().round(1), cam[:, 2].max().round(1))
for c, C in sorted(S['cells'].items()):
    cc = np.array([S['cam_centres_graph'][str(f['k'])] for f in C['frames']]); print(f'{c}: cameras x {cc[:, 0].min():.1f}..{cc[:, 0].max():.1f}, z {cc[:, 2].min():.1f}..{cc[:, 2].max():.1f}')
pl = [json.loads(l) for l in open('/mnt/shared/overnight_20261112/JS1_twin_ledger/records/plants_apr.jsonl')]; pl = [e for e in pl if e['in_vocab']]
near, far, n2 = 0, 0, 0; rows = []
for e in pl:
    pc = {c: x for c, x in (e.get('b_per_cell') or {}).items() if x.get('pos')}
    a = anc.get(e['address'])
    if a is None: continue
    d = {c: float(np.hypot(x['pos'][0] - a[0], x['pos'][2] - a[2])) for c, x in pc.items()}
    if len(d) >= 2: n2 += 1; rows.append((e['address'], {c: round(v, 2) for c, v in d.items()}, {c: pc[c]['n_obs_kf'] for c in pc}))
    if e['b'].get('pos'): dd = float(np.hypot(e['b']['pos'][0] - a[0], e['b']['pos'][2] - a[2])); near += dd <= 0.5; far += dd > 2
print('pooled April record vs its anchor: within 0.5 m', near, 'beyond 2 m', far, 'of', sum(1 for e in pl if e['b'].get('pos')))
print('addresses read in >= 2 chunks:', n2); [print(r) for r in rows[:15]]
dmin = [min(r[1].values()) for r in rows]; dmax = [max(r[1].values()) for r in rows]
print('per address, nearest chunk reading to the anchor: median', np.median(dmin).round(3), '; farthest chunk reading: median', np.median(dmax).round(2))
