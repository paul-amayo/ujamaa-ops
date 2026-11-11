#!/usr/bin/env python3
"""azalai_h1p_dump_changed.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3), design Section 8: a readable dump of the
traces of every instruction whose outcome (frozen gold) differs between v1 and a paraphrase set, in either arm, with the three
wordings of both arms side by side, for reading by eye. Uses the recount's per-instruction outcomes (azalai_h1p_recount.py).
usage: python3 azalai_h1p_dump_changed.py <dir> <recount.json> <out.txt>"""
import sys, json
D, RC, OUT = sys.argv[1:4]
X = json.load(open(RC)); G = json.load(open(f'{D}/instruction_bank_v0_gold.json'))
PRE = {'v1': 'h1v1_seed0', 'A': 'h1p_setA', 'B': 'h1p_setB'}
REC = {k: {a: {r['id']: r for r in map(json.loads, open(f'{D}/{p}_{a}.jsonl'))} for a in ('twin', 'enumerated')} for k, p in PRE.items()}
L = []
for i, p in X['per_instruction'].items():
    arms = [a for a in ('twin', 'enumerated') if len({p[a]['v1'], p[a]['A'], p[a]['B']}) > 1]
    if not arms:
        continue
    g = G[i]; gs = 'REJECT' if g['reject'] else ('any of ' if g['any'] else '') + (str(g['targets']) if len(g['targets']) <= 12 else f"{len(g['targets'])} ids")
    L.append(f"\n################ {i} ({g['category']}) gold: {gs}   outcome changes in: {', '.join(arms)}")
    for a in ('twin', 'enumerated'):
        L.append(f"  [{a}] outcomes v1={p[a]['v1']} A={p[a]['A']} B={p[a]['B']}")
        for k in ('v1', 'A', 'B'):
            r = REC[k][a][i]; f = r.get('final') or {}
            L.append(f"   -- {k}: {r['instruction']}")
            for st in r['steps']:
                if st.get('parsed') == 'tool':
                    rs = json.dumps(st['result']); L.append(f"      CALL {st['tool']} {json.dumps(st['args'])} -> {rs if len(rs) < 300 else rs[:300] + ' ...'}")
                else:
                    L.append(f"      REPLY {st['raw'][:300]}")
            t = f.get('targets', []); L.append(f"      FINAL targets={t if len(t) <= 12 else str(len(t)) + ' ids'} answer={f.get('answer')} reject={f.get('reject')} status={r['status']}")
open(OUT, 'w').write('\n'.join(L) + '\n')
print('wrote', OUT, sum(1 for x in L if x.startswith('\n####')), 'instructions')
