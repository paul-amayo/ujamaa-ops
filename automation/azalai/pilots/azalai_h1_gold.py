#!/usr/bin/env python3
"""azalai_h1_gold.py (29 Oct 2026) — gold targets for the instruction bank v0 from registry_v0.json, one resolver per instruction id.
A gold entry is {"targets": [ids...], "any": bool, "reject": bool, "unresolvable": reason}. 'any' = any one of the targets is correct.
Instructions whose rule needs something the logs do not hold (landmarks, a session history, two robots, a third epoch) are marked
unresolvable and excluded from the score; the count is reported. Run from the pilots dir: python3 azalai_h1_gold.py"""
import json, csv, numpy as np
from pathlib import Path
PUB = Path.home() / 'Documents/UCT/SIL/Publications/Azalai_RSS/numbers'; R = json.load(open(PUB / 'registry_v0.json')); bank = list(csv.DictReader(open(PUB / 'instruction_bank_v0.csv')))
bags = [b for b in R['bags'] if isinstance(b['id'], int)]; byid = {b['id']: b for b in R['bags']}
def row(r): return sorted([b for b in bags if b['row'] == r], key=lambda b: b['n_in_row'])
def nth(r, n): L = row(r); return [L[n - 1]['id']] if 0 < n <= len(L) else []
def last(r, k=1): L = row(r); return [L[-k]['id']] if len(L) >= k else []
def nearest_in_row(r, x, z): L = row(r); return [min(L, key=lambda b: (b['x'] - x) ** 2 + (b['z'] - z) ** 2)['id']] if L else []
def nbr(i, step): b = byid[i]; return nth(b['row'], b['n_in_row'] + step)
def tallest(r): L = [b for b in row(r) if b['height_dec_m'] is not None]; return [max(L, key=lambda b: b['height_dec_m'])['id']] if L else []
def growth(r, g): return [b['id'] for b in row(r) if b['growth'] == g]
def status(st, r=None): return [b['id'] for b in R['bags'] if b['status'] == st and (r is None or b['row'] == r)]
F = R['fruit']; fn = {k: v for k, v in F['fruitnerf']['ours'].items() if isinstance(v, (int, float))}; fuji = F.get('fuji') or {}
def T(ids, any_=False): return dict(targets=ids, any=any_, reject=False)
REJ = dict(targets=[], any=False, reject=True)
def UN(why): return dict(targets=[], any=False, reject=False, unresolvable=why)
HALF = {r: (row(r)[0]['z'] + row(r)[-1]['z']) / 2 for r in range(1, R['rows'] + 1) if row(r)}
gold = {
 'DIR-01': T([412]), 'DIR-02': T(nth(7, 12) + nth(7, 13) + nth(7, 14)), 'DIR-03': T(nth(4, 4)), 'DIR-04': T(nth(2, 5)), 'DIR-05': T(nth(9, 1)),
 'DIR-06': T([118]), 'DIR-07': T([40, 41]), 'DIR-08': T(last(10)), 'DIR-09': T(nth(6, 20)), 'DIR-10': T(nth(5, 30)),
 'UNA-01': T(nth(3, 1)), 'UNA-02': T(last(8)), 'UNA-03': UN('north: site frame heading not in the registry'), 'UNA-04': UN('tunnel entrance: landmark not in the registry'),
 'UNA-05': T(nearest_in_row(6, row(6)[len(row(6)) // 2]['x'], HALF[6]) if row(6) else []), 'UNA-06': T([b['id'] for b in row(R['rows'])], any_=True) if row(R['rows']) else UN('no bags'),
 'UNA-07': T([max(bags, key=lambda b: b['z'])['id']]), 'UNA-08': UN('session history'), 'UNA-09': UN('lane between rows: keyframe target, not a bag'), 'UNA-10': UN('current lane: session state'),
 'COR-01': T([412] + nbr(412, 1)), 'COR-02': T(nth(2, 30) + (nearest_in_row(3, byid[nth(2, 30)[0]]['x'], byid[nth(2, 30)[0]]['z']) if nth(2, 30) else [])),
 'COR-03': T(nth(7, 1) + nth(7, 2)), 'COR-04': T([118] + nbr(118, 1) + [118]), 'COR-05': T(tallest(5)), 'COR-06': UN('any row-1 bag then its successor: open set'),
 'COR-07': T([199, 200, 201]), 'COR-08': T(nth(4, 2) + (nearest_in_row(5, byid[nth(4, 2)[0]]['x'], byid[nth(4, 2)[0]]['z']) if nth(4, 2) else [])), 'COR-09': UN('session history'),
 'COR-10': T([77] + nearest_in_row(8, byid[77]['x'], byid[77]['z'])),
 'SPA-01': T([min(row(4), key=lambda b: abs(b['x'] - np.mean([c['x'] for c in row(5)])))['id']]) if row(4) and row(5) else UN('rows'), 'SPA-02': UN('tunnel entrance: landmark'),
 'SPA-03': T(nbr(412, -1)), 'SPA-04': T([51]), 'SPA-05': T(nbr(300, 1)), 'SPA-06': T(nearest_in_row(8, byid[nth(7, 20)[0]]['x'], byid[nth(7, 20)[0]]['z']) if nth(7, 20) else []),
 'SPA-07': UN('water tank: landmark'), 'SPA-08': T(last(2)), 'SPA-09': T(nearest_in_row(9, row(9)[0]['x'], HALF[9]) if row(9) else []), 'SPA-10': T(last(6, 2)),
 'REG-01': T([b['id'] for b in bags if b['row'] >= 6], any_=True), 'REG-02': UN('one target per side: two-target protocol'), 'REG-03': T([b['id'] for b in bags if b['row'] in (1, 2, 3)], any_=True),
 'REG-04': T([b['id'] for b in row(4) if b['z'] < HALF[4]], any_=True) if row(4) else UN('row'), 'REG-05': T([b['id'] for b in bags if b['z'] > np.percentile([c['z'] for c in bags], 90)], any_=True),
 'REG-06': T([b['id'] for b in row(10)]), 'REG-07': T([b['id'] for b in bags if b['row'] <= 5], any_=True), 'REG-08': T([b['id'] for b in bags if b['row'] in (1, 2)], any_=True),
 'REG-09': T([b['id'] for b in bags if b['row'] in (5, 6)], any_=True), 'REG-10': T([max(bags, key=lambda b: b['x'] + b['z'])['id']]),
 'AMB-01': UN('two robots'), 'AMB-02': UN('two robots'), 'AMB-03': UN('two robots'), 'AMB-04': UN('two robots'), 'AMB-05': UN('two robots'), 'AMB-06': UN('two robots'), 'AMB-07': UN('session history'), 'AMB-08': UN('two robots'), 'AMB-09': UN('two robots'), 'AMB-10': UN('two robots'),
 'FAI-01': REJ, 'FAI-02': REJ, 'FAI-03': REJ, 'FAI-04': REJ, 'FAI-05': REJ, 'FAI-06': REJ, 'FAI-07': REJ, 'FAI-08': REJ, 'FAI-09': REJ, 'FAI-10': REJ,
 'IDE-01': T([max(fn, key=fn.get)]), 'IDE-02': T([min(fn, key=fn.get)]), 'IDE-03': T([t for t, c in fn.items() if c > 200], any_=True), 'IDE-04': UN('tree extent not in the registry'), 'IDE-05': UN('fruit side: not in the registry'),
 'IDE-06': UN('part level: extension'), 'IDE-07': T([t for t, c in fn.items() if c == 0], any_=True) if any(c == 0 for c in fn.values()) else REJ, 'IDE-08': UN('fruit level of a numbered tree: citrus 05 registry not built'), 'IDE-09': UN('adjacency of FruitNeRF trees unknown'), 'IDE-10': T(tallest(7)),
 'COU-01': T([fn.get('tree_03')]) if 'tree_03' in fn else UN('tree'), 'COU-02': UN('citrus 05 rows not in the registry'), 'COU-03': UN('rows of fruit not in the registry'), 'COU-04': T([sum(1 for c in fn.values() if c > 150)]),
 'COU-05': T([len(row(5))]), 'COU-06': T([len(status('removed')) + len(status('unclear'))]), 'COU-07': UN('citrus rows'), 'COU-08': UN('citrus rows'), 'COU-09': T([sum(1 for b in bags if b['growth'] == 'grown')]), 'COU-10': T([R['rows']]),
 'CHA-01': T(status('removed') + status('unclear'), any_=True), 'CHA-02': T(status('added')), 'CHA-03': T(growth(7, 'stable'), any_=True), 'CHA-04': T([min([b for b in bags if b['dh_apr_m'] is not None], key=lambda b: b['dh_apr_m'])['id']]),
 'CHA-05': UN('moved: positional pairs beyond 0.5 m are unpaired, not moved'), 'CHA-06': T(status('removed')), 'CHA-07': T(status('removed', 3) + status('added') if False else [b['id'] for b in R['bags'] if b['row'] == 3 and b['status'] in ('removed', 'added', 'unclear')]),
 'CHA-08': T([max([b for b in row(2) if b['dh_apr_m'] is not None], key=lambda b: b['dh_apr_m'])['id']]) if [b for b in row(2) if b['dh_apr_m'] is not None] else UN('row 2'), 'CHA-09': T(status('unclear')), 'CHA-10': REJ}
out = {}
for it in bank:
    g = gold.get(it['id'], UN('no resolver')); out[it['id']] = dict(instruction=it['instruction'], category=it['category'], **g)
json.dump(out, open(PUB / 'instruction_bank_v0_gold.json', 'w'), indent=1)
from collections import Counter
print('resolvable', sum(1 for g in out.values() if 'unresolvable' not in g), 'of', len(out)); print(Counter((g['category'], 'unresolvable' in g) for g in out.values()))
