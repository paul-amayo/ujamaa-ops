#!/usr/bin/env python3
"""azalai_h1p_gold_corrected.py (11 Nov 2026) — H1 paraphrase test (AZ-3), corrected-gold SENSITIVITY (not adopted; the frozen
gold instruction_bank_v0_gold.json is read and never written). Builds a corrected copy that changes the gold for the issues of
H1_v1_results.md that can be changed from the bank's own rule text, one documented change per issue, and leaves the two that need
a judgement about what the instruction means (CHA-07, COU-06), for which labelled alternative readings are written beside it.

  fix 1  co-reference ids: the gold builder's COR-nn resolvers are entered under the bank's ids CO--nn (8 of 10 have one)
  fix 2  CHA-09: the gold list is marked order_free (the instruction names a set; the rule text gives no order)
  fix 5  eleven rows: the ten-row halves and middle of the rule text are carried to eleven rows about the centre row 6
         (REG-07 left block rows 1-6; REG-09 middle rows 5-7; REG-01 right side is already rows 6-11 in the frozen gold)
  left   CHA-07 (issue 3) and COU-06 (issue 4): unchanged in the corrected copy; readings in the 'readings' block

Writes <out>/h1p_gold_corrected_sensitivity.json (same form as the frozen gold, plus the key order_free on CHA-09) and
<out>/h1p_gold_corrected_changes.json (every change with before / after, and the alternative readings as overrides).
usage: python3 azalai_h1p_gold_corrected.py <Azalai_RSS/numbers> <out dir>"""
import sys, json, copy, hashlib
num, out = sys.argv[1:3]
G0 = json.load(open(f'{num}/instruction_bank_v0_gold.json')); R = json.load(open(f'{num}/registry_v0.json'))
bags = [b for b in R['bags'] if isinstance(b['id'], int)]; byid = {b['id']: b for b in bags}
def row(r): return sorted([b for b in bags if b['row'] == r], key=lambda b: b['n_in_row'])
def nth(r, n): L = row(r); return [L[n - 1]['id']] if 0 < n <= len(L) else []
def near(r, i): b = byid[i]; return [min(row(r), key=lambda c: (c['x'] - b['x']) ** 2 + (c['z'] - b['z']) ** 2)['id']]
def nbr(i, s): b = byid[i]; return nth(b['row'], b['n_in_row'] + s)
def tallest(r): L = [b for b in row(r) if b['height_dec_m'] is not None]; return [max(L, key=lambda b: b['height_dec_m'])['id']]
# the gold builder's COR resolvers (azalai_h1_gold.py, lines 'COR-01'..'COR-10'), as azalai_h1_score_v1.py evaluates them
COR = {'CO--01': [412] + nbr(412, 1), 'CO--02': nth(2, 30) + near(3, nth(2, 30)[0]), 'CO--03': nth(7, 1) + nth(7, 2), 'CO--04': [118] + nbr(118, 1) + [118], 'CO--05': tallest(5),
       'CO--07': [199, 200, 201], 'CO--08': nth(4, 2) + near(5, nth(4, 2)[0]), 'CO--10': [77] + near(8, 77)}
assert R['rows'] == 11
G = copy.deepcopy(G0); changes = []


def change(issue, i, why, **new):
    before = {k: G[i].get(k) for k in ('targets', 'any', 'reject', 'unresolvable', 'order_free') if k in G[i] or k in new}
    for k, v in new.items():
        if v is None: G[i].pop(k, None)
        else: G[i][k] = v
    def short(d): return {k: (v if not isinstance(v, list) or len(v) <= 12 else f'{len(v)} ids') for k, v in d.items()}
    changes.append(dict(issue=issue, id=i, instruction=G[i]['instruction'], rule_text=why, before=short(before), after=short({k: G[i].get(k) for k in before})))


for i, t in COR.items():
    change('1 co-reference ids', i, 'the builder holds a resolver for this instruction under COR-nn; the bank id is CO--nn, so the frozen gold says "no resolver". The resolver is entered under the bank id, unchanged.', targets=t, any=False, reject=False, unresolvable=None)
change('2 CHA-09 order', 'CHA-09', 'rule text "JIRS vetting: unclear candidates": a set, no order stated (rules that mean an order say so: DIR-02, DIR-07, REG-03, REG-06 "in order").', order_free=True)
change('5 eleven rows', 'REG-07', 'rule text "rows 1-5" is the lower half of ten rows; with eleven rows the centre row 6 lies in neither half, and is accepted on either side (the frozen gold already accepts it on the right: REG-01 = rows 6-11).', targets=[b['id'] for b in bags if b['row'] <= 6])
change('5 eleven rows', 'REG-09', 'rule text "rows 5-6" is the two central rows of ten; with eleven rows the centre is row 6, and the central rows are 5-7.', targets=[b['id'] for b in bags if b['row'] in (5, 6, 7)])
assert sorted(G0['REG-01']['targets']) == sorted(b['id'] for b in bags if b['row'] >= 6)          # issue 5 needs no change in REG-01
assert G0['CHA-07']['targets'] == [] and G0['COU-06']['targets'] == [5]
# alternative readings (never part of the corrected copy): overrides the scorer applies one at a time
st = lambda s: [b['id'] for b in R['bags'] if b['status'] == s]
readings = {
 'CHA-07 unresolvable': dict(id='CHA-07', note='the rule text needs "moved", which the builder marks as not held by the registry (CHA-05): the instruction is taken out of the score', override=dict(unresolvable='reading: rule needs "moved", not in the registry')),
 'CHA-07 reject also accepted': dict(id='CHA-07', note='a refusal is accepted beside the empty answer', override=dict(reject_also_accepted=True)),
 'COU-06 = 1 (removed only)': dict(id='COU-06', note="the registry's own word for gone in April: status removed", override=dict(targets=[len(st('removed'))])),
 'COU-06 = 1 or 5': dict(id='COU-06', note='removed, or removed + unclear', override=dict(count_also=[len(st('removed'))])),
 'eleven rows, strict halves': dict(id='REG-01,REG-07,REG-09', note='the centre row 6 in neither half: right side rows 7-11, left block rows 1-5, middle rows = row 6 only',
                                    override={'REG-01': dict(targets=[b['id'] for b in bags if b['row'] >= 7]), 'REG-07': dict(targets=[b['id'] for b in bags if b['row'] <= 5]), 'REG-09': dict(targets=[b['id'] for b in bags if b['row'] == 6])}),
}
json.dump(G, open(f'{out}/h1p_gold_corrected_sensitivity.json', 'w'), indent=1)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(label='SENSITIVITY, not adopted: the gold of record is instruction_bank_v0_gold.json', frozen_gold_sha256=sha(f'{num}/instruction_bank_v0_gold.json'), registry_sha256=sha(f'{num}/registry_v0.json'),
               n_resolvable_frozen=sum(1 for g in G0.values() if 'unresolvable' not in g), n_resolvable_corrected=sum(1 for g in G.values() if 'unresolvable' not in g),
               changes=changes, left_unchanged={'CHA-07': 'issue 3: the gold [] follows from the rule text for the part the registry holds (no row 3 bag is removed or added); whether a refusal should also score depends on whether "changed" includes movement the registry does not hold: a judgement about meaning',
                                                'COU-06': 'issue 4: the rule text is len(unmatched_A); the registry holds the vetted status instead, and "missing" can be read as removed (1), removed + unclear (5, the builder) or every December-only candidate: a judgement about meaning'},
               readings=readings), open(f'{out}/h1p_gold_corrected_changes.json', 'w'), indent=1)
for c in changes: print(c['issue'], c['id'], c['before'], '->', c['after'])
print('resolvable', sum(1 for g in G0.values() if 'unresolvable' not in g), '->', sum(1 for g in G.values() if 'unresolvable' not in g))
