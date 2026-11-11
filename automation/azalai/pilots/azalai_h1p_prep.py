#!/usr/bin/env python3
"""azalai_h1p_prep.py (11 Nov 2026) — H1 paraphrase test (AZ-3): from the committed paraphrase table
(Azalai_RSS/numbers/instruction_bank_v0_paraphrases.csv) write the bank files the v1 harness reads with --bank (same columns as
the frozen bank: id, category, site, instruction, target_rule; the id is the bank's id, the instruction is the paraphrase):

  h1p_bank_setA.csv, h1p_bank_setB.csv            the 61 resolvable instructions, in bank order (the primary runs)
  h1p_bank_setA_change.csv, h1p_bank_setB_change.csv   the 9 change instructions of each set (the determinism re-run)
  h1p_bank_corA.csv, h1p_bank_corB.csv            the 8 co-reference instructions with a builder resolver (supplementary block)

It reads no record and no score. Design of record: Azalai_RSS/numbers/pilots_20261111/H1_paraphrase_design.md.
usage: python3 azalai_h1p_prep.py <Azalai_RSS/numbers> <out dir>"""
import sys, csv, json
num, out = sys.argv[1:3]
bank = list(csv.DictReader(open(f'{num}/instruction_bank_v0.csv'))); G = json.load(open(f'{num}/instruction_bank_v0_gold.json'))
P = list(csv.DictReader(open(f'{num}/instruction_bank_v0_paraphrases.csv')))
res = [b['id'] for b in bank if 'unresolvable' not in G[b['id']]]; assert len(res) == 61
by = {(p['id'], p['set']): p for p in P}; assert len(by) == len(P)
for p in P:                                             # the table must still agree with the frozen bank, word for word
    b = next(x for x in bank if x['id'] == p['id']); assert (p['original'], p['category'], p['site'], p['target_rule']) == (b['instruction'], b['category'], b['site'], b['target_rule']), p['id']
    assert p['paraphrase'].strip() and p['paraphrase'] != p['original']
prim = [p['id'] for p in P if p['block'] == 'primary' and p['set'] == 'A']; cor = [p['id'] for p in P if p['block'] != 'primary' and p['set'] == 'A']
assert prim == res and len(cor) == 8 and all(G[i]['category'] == 'co-reference resolution' for i in cor)


def write(name, ids, s):
    with open(f'{out}/{name}', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['id', 'category', 'site', 'instruction', 'target_rule'])
        for i in ids: p = by[(i, s)]; w.writerow([i, p['category'], p['site'], p['paraphrase'], p['target_rule']])
    print(name, len(ids))


for s in 'AB':
    write(f'h1p_bank_set{s}.csv', res, s); write(f'h1p_bank_set{s}_change.csv', [i for i in res if G[i]['category'] == 'change'], s); write(f'h1p_bank_cor{s}.csv', cor, s)
